from torch.utils.data import Dataset
import torch

class GraphicalDataset(Dataset):
    """
    A dataset combined with a graphical model.
    """
    def __init__(self, config, is_test):
        super(Dataset, self).__init__()
        self.config = config
        self.length = config.data.test_dataset_length if is_test else config.data.dataset_length
        self.finite_length = config.data.finite_length
        self.is_test = is_test
        self.fit_intermediate = self.config.data.fit_intermediate

    def avg_log_prob(self, N=None):
        return torch.tensor(0.0)

    def condition_batch(self, batch, obs_mask):
        return batch, obs_mask

    def __len__(self):
        return self.length

    def plot(self, samples_cont, samples_disc, obs_mask, **kwargs):
        return None, None # fig, ax
