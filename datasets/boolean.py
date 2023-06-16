import itertools as it
import torch
import numpy as np
from numpy.random import randint
from .graphical_dataset import GraphicalDataset
import matplotlib.pyplot as plt
from .utils import RNG


class Boolean(GraphicalDataset):

    def __init__(self, config, is_test):
        super().__init__(config, is_test)
        self.n = self.config.data.n
        if config.data.vary_dimensions:
            self.n_max = self.config.data.max_n
        self.generated_items = 0

        # randomly sample link functions
        with RNG(1):
            self.layer_funcs = {}
            for layer in range(self.n, 0, -1):
                self.layer_funcs[layer] = []
                for i in range(0, 2**(self.n-1)):
                    self.layer_funcs[layer].append(np.random.choice(['or', 'and']))
            # if a hidden activation is always one or the other value, consider changing the link function
            actual_fit_intermediate = self.fit_intermediate
            self.fit_intermediate = True
            N = 1000
            def get_freqs():
                return sum(self[i][1] for i in range(N))
            # print('before rejig')
            # print(self.layer_funcs)
            # print(get_freqs())
            for layer in range(self.n-1, 0, -1):
                freqs = get_freqs()
                layer_output_start = sum(2**l for l in range(self.n, layer-1, -1))
                layer_n_outputs = 2**(layer-1)
                layer_output_freqs = freqs[layer_output_start:layer_output_start+layer_n_outputs]
                for i, freq in enumerate(layer_output_freqs):
                    # swap out layer funcs that may be contributing to a near-constant output
                    if freq > 0.8*N and (self.layer_funcs[layer][i] == 'or'):
                        self.layer_funcs[layer][i] = 'and'
                    elif freq < 0.2*N and (self.layer_funcs[layer][i] == 'and'):
                        self.layer_funcs[layer][i] = 'or'
            # print('after rejig')
            # print(self.layer_funcs)
            # print(get_freqs())
            self.fit_intermediate = actual_fit_intermediate


    def run(self, input):
        def apply(func, ins):
            return {'and': ins[0]*ins[1],
                    'or': (ins[0]+ins[1]).clip(max=1)}[func]
        activations = []
        layer_input = input
        for layer in range(self.n, 0, -1):
            funcs = self.layer_funcs[layer]
            assert len(layer_input) == 2**layer
            #print(layer_input)
            layer_out = [apply(func, layer_input[i:i+2]) for i, func in zip(range(0, len(layer_input), 2), funcs)]
            #print(layer_out)
            if self.fit_intermediate or layer == 1:
                activations.append(layer_out)
            layer_input = layer_out
        return activations

    @property
    def intermediate_mask(self):
        assert self.fit_intermediate
        disc = torch.cat([
            torch.zeros(2**self.n),
            torch.ones(sum(2**l for l in range(1, self.n))),
            torch.zeros(1),
        ], dim=0)
        cont = torch.zeros(0)
        disc = disc.repeat_interleave(2, dim=0)
        return torch.cat([cont, disc], dim=0).view(1, -1)

    @property
    def n_cont(self):
        return 0

    @property
    def shared_var_embeds(self):
        embeds = []
        for i, layer in enumerate(range(self.n, 0, -1)):
            input_size = 2**layer
            embeds.extend([i]*input_size)
        embeds.append(embeds[-1]+1)
        return embeds

    def vary_dimensions(self):
        if self.config.data.vary_dimensions:
            self.n = np.random.randint(1, self.n_max+1)

    def set_dims_to_max(self):
        self.n = self.n_max

    def __getitem__(self, index):
        if self.finite_length:
            seed = index + 2**31 if self.is_test else index
            with RNG(seed):
                return self.__unseeded_getitem__(index)
        else:
            return self.__unseeded_getitem__(index)

    def __unseeded_getitem__(self, index):
        # first batch is always at maximum initial size
        if self.config.data.vary_dimensions and index % self.config.training.batch_size == 0 and index > 0:
            self.vary_dimensions()
        self.generated_items += 1
        cont = torch.zeros(0)
        input = np.random.choice([0, 1], size=2**self.n)
        activations = self.run(input)
        disc = torch.cat([torch.tensor(input).long()]+[torch.tensor(a).long() for a in activations], dim=0)
        return cont, disc, (self.n_cont, self.n_discrete_options, self.shared_var_embeds), (self.n,)

    @property
    def n_discrete_options(self):
        n_inputs = 2**self.n
        return [2]*((2*n_inputs-1) if self.fit_intermediate else (n_inputs+1))

    def faithful_inversion_edges(self):
        edges = set()
        if not self.fit_intermediate:
            for input in range(2**self.n):
                edges.add((input, 2**self.n))   
                edges.add((2**self.n, input))
            return edges
        output_start_idx = 0
        for layer in range(self.n, 0, -1):
            input_start_idx = output_start_idx
            input_size = 2**layer
            output_start_idx = input_start_idx + input_size
            for i in range(0, input_size, 2):
                input1_idx = input_start_idx + i
                input2_idx = input_start_idx + i + 1
                output_index = output_start_idx + i//2
                for edge in [(input1_idx, output_index), (input2_idx, output_index),
                              (output_index, input1_idx), (output_index, input2_idx)]:
                    edges.add(edge)
        return edges

    def validation_metrics(self, samples_disc, samples_cont, **kwargs):
        batch_inputs = samples_disc[:, :2**self.n]
        batch_output = samples_disc[:, -1]
        # accuracy of final prediction
        correct = 0
        balanced_correct = 0
        all_targets = torch.tensor([self.run(input)[-1][0] for input in batch_inputs])
        for input, output in zip(batch_inputs, batch_output):
            activations = self.run(input)
            target = activations[-1][0]
            correct += (target == output)
            prop_with_target = (target == all_targets).float().mean()
            balanced_correct += (target == output).float() / (2*prop_with_target)
        # accuracy of all intermediate predictions (if they exist)
        prop_intermediate_correct = 0
        prop_all_intermediate_correct = 0
        balanced_prop_intermediate_correct = 0
        balanced_prop_all_intermediate_correct = 0
        all_activations = torch.stack([torch.tensor(sum(self.run(input), [])) for input in batch_inputs])
        for input, sample_disc in zip(batch_inputs, samples_disc):
            activations = torch.tensor(sum(self.run(input), []))
            accurate = (activations == sample_disc[2**self.n:].cpu()).float()
            prop_intermediate_correct += accurate.mean()
            prop_all_intermediate_correct += accurate.all()
            prop_with_target = (activations.reshape(1, -1) == all_activations).float().mean(dim=0)
            balanced_accurate = accurate / (2*prop_with_target)
            balanced_prop_intermediate_correct += balanced_accurate.mean()
            balanced_prop_all_intermediate_correct += balanced_accurate.all()

        return {
            'accuracy': correct/len(batch_inputs),
            'intermediate_accuracy': prop_intermediate_correct/len(batch_inputs),
            'strict_intermediate_accuracy': prop_all_intermediate_correct/len(batch_inputs),
            'balanced_accuracy': balanced_correct/len(batch_inputs),
            'balanced_intermediate_accuracy': balanced_prop_intermediate_correct/len(batch_inputs),
            'balanced_strict_intermediate_accuracy': balanced_prop_all_intermediate_correct/len(batch_inputs),
        }

    def set_dims(self, dims):
        self.n = dims[0]

    def sample_obs_mask(self, B, device, obs_prop=None, exclude_mask=None):
        input_size = 2**self.n
        emb = torch.zeros((B, len(self.n_discrete_options), 1), device=device)   #  A, E, R
        emb[:, :input_size] = 1.
        xt = emb.repeat_interleave(2, dim=1).squeeze(dim=-1)
        return {"emb": emb, "xt": xt}

    def plot(self, samples_cont, samples_disc, obs_mask, **kwargs):
        B = len(samples_cont)
        fig, axes = plt.subplots(nrows=B, ncols=4)
        return fig, np.array(axes).reshape(-1)[0]