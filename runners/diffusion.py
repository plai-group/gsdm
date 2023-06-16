import os
import logging
import time
import glob
import itertools

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import TransformerModel
from models.ema import EMAHelper
from functions import get_optimizer
from functions.utils import DiffusionProcess
from functions.losses import loss_registry
from datasets import get_dataset
from functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu
import wandb
import matplotlib.pyplot as plt

from functions.denoising import ddpm_elbo

from runners.vaeac.VAEAC import VAEAC
from runners.vaeac.imputation_networks import get_imputation_networks



def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config

        assert config.training.batch_size >= config.sampling.sampling_batch_size

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        self.dataset = dataset
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        valid_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.sampling_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        self.valid_batch = next(iter(valid_loader))

        wandb.log({'optimal_log_prob': dataset.avg_log_prob()})

        if args.vaeac:
            one_hot_max_sizes =  [0]*dataset.n_cont + dataset.n_discrete_options
            self.networks = get_imputation_networks(one_hot_max_sizes, width=args.vaeac_width, depth=args.vaeac_depth, latent_dim=args.vaeac_latent_dim)
            model = VAEAC(
                self.networks['reconstruction_log_prob'],
                self.networks['proposal_network'],
                self.networks['prior_network'],
                self.networks['generative_network']
            )
            assert not config.model.ema
        model = TransformerModel(config, dataset,
                                 dataset.faithful_inversion_edges())
        print('Parameters:', sum(p.numel() for p in model.parameters()))
        print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        for name, module in model.named_children():
            print(name, sum(p.numel() for p in module.parameters())/1e6, 'million')


        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        if self.args.weight_loss:
            loss_weights = DiffusionProcess(self.betas, self.num_timesteps).elbo_weighting
            wandb.log({'log_loss_weights': wandb.Histogram(loss_weights.cpu().log())})
        else:
            loss_weights = None
        for epoch in range(start_epoch, self.config.training.n_epochs):
            epoch_start = time.time()
            data_start = time.time()
            data_time = 0
            repeating_train_loader = itertools.cycle(train_loader)
            for i, batch in zip(range(config.training.max_epoch_iters), repeating_train_loader):
                data_time += time.time() - data_start
                model.train()
                step += 1

                data_dims = None
                if self.config.data.vary_dimensions:
                    data_dims = [dim[0] for dim in batch[3]]
                    dataset.set_dims(data_dims)
                    model_dims = batch[2]
                    model.module.reset_dimensions(model_dims, data_dims, log_stats=(i % self.args.log_freq == 0))

                # VAEAC expects data to just be a float array, with discrete values represented as integers
                if args.vaeac:
                    dequant_log_prob = torch.tensor(0.)
                    cont, disc = batch[0], batch[1]
                    x = torch.cat([cont.to(torch.float32), disc.to(torch.float32)], dim=1)
                else:
                    x, dequant_log_prob = model.module.dequantize(batch[0:2])
                    dequant_log_prob = dequant_log_prob.mean()

                n = x.size(0)
                x = x.to(self.device)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                obs_mask = dataset.sample_obs_mask(len(x), self.device)

                if args.just_save_data_to != "":
                    # save groundtruth data
                    with open(args.just_save_data_to+'_groundtruth.tsv', 'a') as fg, open(args.just_save_data_to+'_train.tsv', 'a') as ft:
                        disc = batch[1].cpu().numpy()
                        is_obs = obs_mask['emb'].cpu().numpy()
                        for row, obs_row in zip(disc, is_obs):
                            fg.write('\t'.join(str(el) for el in row) + '\n')
                            ft.write('\t'.join(str(el) if o else 'nan' for el, o in zip(row, obs_row)) + '\n')
                    continue

                if self.config.data.supervise_intermediate:
                    loss_mask = None
                else:
                    mask = dataset.intermediate_mask
                    loss_mask = (1 - mask).to(self.device)
                    x = x * loss_mask
                    e = e * loss_mask

                if args.vaeac:
                    mask = obs_mask["emb"].squeeze(2)
                    loss = -model.module.batch_vlb(x, 1-mask)
                    loss = loss.sum() / x.numel()
                    mse_loss = 0.
                else:
                    mse_loss = loss_registry[config.model.type](model, x, t, e, b, w=loss_weights,
                                                                predict=self.config.model.predict,
                                                                obs_mask=obs_mask,
                                                                log_attn=False,
                                                                mean_over_latents=config.training.mean_latents_loss,
                                                                loss_mask=loss_mask,
                                                                regression=self.args.regression)
                    loss = mse_loss - dequant_log_prob.to(mse_loss.device)
                do_attn_reg = self.config.model.attn_reg_lambda != 0 and i+epoch*config.training.max_epoch_iters < config.training.attn_reg_iters

                if do_attn_reg:
                    loss = -model.module.attn_reward  # only use attn reward for first 1000 iters

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                if self.args.vaeac:
                    grad_norm = 0.
                else:
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if i % self.args.log_freq == 0:
                    wandb.log({'loss': loss, 'epoch': epoch, 'iter': i, 'dequant_log_prob': dequant_log_prob,
                               'mse_loss': mse_loss, 'grad_norm': grad_norm})


                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

            if args.just_save_data_to != "":
                continue
            wandb.log({'epoch_time': time.time()-epoch_start})
            self.validate(model=model, ema_helper=ema_helper, iteration=step)

    def validate(self, model, ema_helper, iteration):
        vis_start = time.time()
        # draw samples -----------------------------------
        model.eval()
        if self.config.model.ema:
            valid_model = ema_helper.ema_copy(model)
        else:
            valid_model = model
        valid_model.eval()

        data_dims = None
        if self.config.data.vary_dimensions:
            data_dims = [dim[0] for dim in self.valid_batch[3]]
            self.dataset.set_dims(data_dims)
            model_dims = self.valid_batch[2]
            model.module.reset_dimensions(model_dims, data_dims, plot_mask=True)
            valid_model.module.reset_dimensions(model_dims, data_dims)

        batch_size = self.config.sampling.sampling_batch_size
        obs_props = self.config.data.obs_props if hasattr(self.config.data, 'obs_props') else [0.0,]
        obs_masks = [self.dataset.sample_obs_mask(batch_size, self.device, obs_prop=op)
                     for op in obs_props]
        mask_descs = [f'-obs-{op}' for op in obs_props]


        for obs_prop, obs_mask, desc in zip(obs_props, obs_masks, mask_descs):
            sampling_start_time = time.time()
            self.log_samples(valid_model, self.valid_batch, obs_mask=obs_mask, prefix=desc+'-ema')
            sampling_time = time.time() - sampling_start_time
        # compute ELBO -----------------------------------
        elbo_mask = self.dataset.sample_obs_mask(self.config.sampling.sampling_batch_size, self.device)
        self.log_elbos(valid_model, self.valid_batch, elbo_mask, prefix='valid-ema')
        self.log_elbos(model, self.valid_batch, elbo_mask, prefix='valid')
        # reset state dict -------------------------------
        model.train()
        del valid_model
        wandb.log({'vis_time': time.time()-vis_start,
                   "sampling_time": sampling_time,
                   "iteration": iteration}, commit=False)
        wandb.log({})

    def log_samples(self, model, batch, obs_mask, prefix=''):
        batch, obs_mask = self.dataset.condition_batch(batch, obs_mask)
        if self.args.vaeac:
            cont, disc = batch[0], batch[1]
            gt_cont, gt_disc = batch[0], batch[1]
            x = torch.cat([cont.to(torch.float32), disc.to(torch.float32)], dim=1).to(self.device)
            unobs_mask = 1-obs_mask['emb'].squeeze(2)
            likelihood = model.module.generate_samples_params(x, mask=unobs_mask)
            likelihood = likelihood.squeeze(1)  # get rid of n_samples dimension (I think)
            samples = self.networks["sampler"](likelihood)
            n_cont = self.dataset.n_cont
            samples = samples * unobs_mask + x * (1-unobs_mask)
            samples_cont, samples_disc = samples[:, :n_cont].detach().cpu(), samples[:, n_cont:].detach().cpu()
            samples_disc = samples_disc.to(torch.int64)
        else:
            gt_cont, gt_disc = batch[0], batch[1]
            data = model.module.dequantize(batch[:2])[0].to(self.device)
            xT = model.module.sample_xT(self.config.sampling.sampling_batch_size)
            samples = self.sample(xT, model, obs=data, obs_mask=obs_mask)
            samples_cont, samples_disc = model.module.requantize(samples)

        dataset_metrics = self.dataset.validation_metrics(samples_disc, samples_cont, gt_cont=gt_cont, gt_disc=gt_disc, model=model)

        dataset_metrics = {f'samples{prefix}/{k}': v for k, v in dataset_metrics.items()}
        wandb.log(dataset_metrics, commit=False)
        fig, ax = self.dataset.plot(samples_cont, samples_disc, obs_mask=obs_mask)
        if fig is not None:
            wandb.log({f'samples{prefix}': wandb.Image(fig)}, commit=False)
        if not self.config.sampling.last_only:
            xs, x0_preds = self.sample(xT, model, obs=data,
                                       obs_mask=obs_mask, last=False)
            for i, x0_pred in enumerate(x0_preds):
                samples_cont, samples_disc = model.module.requantize(x0_pred)
                fig, ax = self.dataset.plot(samples_cont, samples_disc, obs_mask=obs_mask)
                fig.savefig(f"plots/{self.config.data.dataset}/x0_pred_{i:04}.png")
                plt.close()

    def log_elbos(self, model, data_batch, obs_mask, prefix=''):
        if self.args.vaeac:
            return
        n_cont = data_batch[0].shape[1]
        x0, dequant_log_prob = model.module.dequantize(data_batch[:2])
        elbos = ddpm_elbo(model, x0=x0, b=self.betas, T=self.num_timesteps, n_cont=n_cont,
                          obs_mask=obs_mask, config=self.config)
        elbos['quantized_elbo'] = elbos['elbo'] - dequant_log_prob.mean()
        elbos = {prefix+'/'+k: v for k, v in elbos.items()}
        wandb.log(elbos, commit=False)

    def evaluate(self):
        self.dataset, self.test_dataset = get_dataset(self.args, self.config)
        model, ema_helper = self.load_ckpt(self.args.eval_path, self.dataset)
        loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.sampling.sampling_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
        )
        self.validate(model, ema_helper, loader)

    def load_ckpt(self, eval_path, dataset):
        model = TransformerModel(self.config, dataset,
                                 dataset.faithful_inversion_edges())
        states = torch.load(
            self.args.eval_path,
            map_location=self.config.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        # remove temporary internal parameters that have mismatching shape
        if self.config.model.impose_sparsity != 'not':
            del states[0]["module.valid_indices_mask"]
            del states[0]["module.attendable_indices"]
            valid_indices_mask = model.module.valid_indices_mask
            attendable_indices = model.module.attendable_indices

            del model.module.valid_indices_mask
            del model.module.attendable_indices

        model.load_state_dict(states[0], strict=False)
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
            if self.config.model.impose_sparsity != 'not':
                ema_helper.valid_indices_mask = valid_indices_mask
                ema_helper.attendable_indices = attendable_indices
        else:
            ema_helper = None

        # reattach mask
        if self.config.model.impose_sparsity != 'not':
            model.module.valid_indices_mask = valid_indices_mask
            model.module.attendable_indices = attendable_indices
        return model, ema_helper

    def sample(self, x, model, obs=None, obs_mask=None, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.regression:
            z = torch.zeros_like(x)
            t = torch.zeros(x.size(0), device=self.device)
            obs = obs * obs_mask["xt"]
            output = model(z, t, log_attn=False, obs=obs, obs_mask=obs_mask)
            output = output * (1-obs_mask["xt"]) + obs * obs_mask["xt"]
            return output.detach().cpu()
        elif self.args.sample_type == "ddim":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddim_steps

            xs = ddim_steps(x, seq, model, self.betas, eta=self.args.eta, config=self.config, predict=self.config.model.predict,
                            obs=obs, obs_mask=obs_mask)
            x = xs
        elif self.args.sample_type == "ddpm":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas, config=self.config,
                           obs=obs, obs_mask=obs_mask)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x
