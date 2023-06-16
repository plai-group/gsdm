import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import wandb

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="runs", help="Path for saving running related data."
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="warning",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--eval_path", type=str, help="If specified, evaluate this ckpt instead of training.")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument( "--resume_training", action="store_true", help="Whether to resume training")
    parser.add_argument( "--wandb_tags", type=str, default=None, help="Tags for wandb", nargs='+')
    parser.add_argument( "--resume_id", type=str, default=None)
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    str2bool = lambda s: 't' in s.lower()
    parser.add_argument(
        "--ni",
        type=str2bool, default=True,
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="ddpm",
        help="sampling approach (ddim or ddpm)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--weight_loss",
        action="store_true",
        help="Apply weights to the MSE loss so that it is equivalent to the ELBO.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--just_save_data_to", type=str, default="", help="If specified, saves data to this .tsv path instead of training"
    )
    parser.add_argument(
        "--log_freq", type=int, default=10000, help="How often to log training progress."
    )
    parser.add_argument(
        "--vaeac", type=str2bool, default=False, help="Train VAEAC instead of a diffusion model."
    )
    parser.add_argument(
        "--regression", type=str2bool, default=False, help="Train a regression loss with MSE instead of a diffusion model."
    )

    parser.add_argument("--vaeac_width", type=int, default=256, help="Width of the VAEAC model.")
    parser.add_argument("--vaeac_depth", type=int, default=10, help="Depth of the VAEAC model.")
    parser.add_argument("--vaeac_latent_dim", type=int, default=64, help="Latent dimension of the VAEAC model.")
    parser.add_argument("--sequence", action="store_true")
    configurable = {'model': {'resnet': str2bool,
                              'num_transformers': int,
                              'emb_dim': int,
                              'predict': str,
                              'softmax': str2bool,
                              'var_embedding': str2bool,
                              'conditional': str,
                              'n_heads': int,
                              'attn_dim_reduce': int,
                              'impose_sparsity': str,
                              'attn_reg_mu': float,
                              'attn_reg_lambda': float,
                              'use_shared_var_embeds': str2bool,
                              'use_shared_var_positions': str2bool,
                              'max_attn_matrix_size': int,
                              'ema': str2bool,},
                    'training': {'n_epochs': int,
                                 'batch_size': int,
                                 'max_epoch_iters': int,
                                 'attn_reg_iters': int,
                                 'validation_freq': int,
                                 "mean_latents_loss": str2bool},
                    'optim': {'lr': float},
                    'data': {'n_discrete_options': eval,
                             'type': int,
                             'n': int,
                             'm': int,
                             't': int,
                             'max_n': int,
                             'max_m': int,
                             'max_t': int,
                             'vary_dimensions': str2bool,
                             'fit_intermediate': str2bool,
                             'save_sparsity_mask': str2bool,
                             'sparsity_mask_index': int,
                             'supervise_intermediate': str2bool,
                             'finite_length': str2bool,
                             'dataset_length': int,
                             'num_workers': int,
                             'nn_option': str,
                             'number_of_network_passes': int,
                             'norm_type': str,
                             },
                    'diffusion': {'beta_schedule': str,
                                  'beta_start': float,
                                  'beta_end': float,
                                  'num_diffusion_timesteps': int},
                    'sampling': {'sampling_batch_size': int,
                                 'fixed_batch': str2bool,
                                 'last_only': str2bool}}

    for arg_type, type_configurable in configurable.items():
        for arg, typ in type_configurable.items():
            parser.add_argument('--'+arg, type=typ, default=None)

    configurable['data']['obs_props'] = [0.0] # default value, add to dict keys
    parser.add_argument('--obs_props', nargs='+', type=float,
                        help='Proportions of variables observed during validation.')

    args = parser.parse_args()
    # parse config file
    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)
    for arg_type, type_configurable in configurable.items():
        for arg in type_configurable:
            if getattr(args, arg) is not None:
                config[arg_type][arg] = getattr(args, arg)
    new_config = dict2namespace(config)
    print(new_config)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

def set_log_path(args, config):
    train = args.eval_path is None
    if train:
        args.log_path = os.path.join(args.exp, "logs", wandb.run.id)
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    try:
        runner = Diffusion(args, config)

        # initialise wandb ----------------------------------
        unpacked_config = {}
        for k, v in config.__dict__.items():
            if isinstance(v, argparse.Namespace):
                inner_dict = {k+'.'+k2: v2 for k2, v2 in v.__dict__.items()}
                unpacked_config = {**unpacked_config, **inner_dict}
            else:
                unpacked_config[k] = v
        wandb.init(entity='universal-conditional-ddpm', project='universal-conditional-ddpm',
                   config={**unpacked_config, **args.__dict__}, id=args.resume_id, tags=args.wandb_tags)
        set_log_path(args, config)
        # ---------------------------------------------------
        if args.eval_path is None:
            logging.info("Writing log file to {}".format(args.log_path))
            logging.info("Exp instance id = {}".format(os.getpid()))
            logging.info("Exp comment = {}".format(args.comment))
            runner.train()
        else:
            runner.evaluate()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
