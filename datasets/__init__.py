import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datasets.bmf import BMF
from datasets.hbmf import HBMF
from datasets.cmf import CMF
from datasets.sudoku import Sudoku
from datasets.sorting import Sorting
from datasets.boolean import Boolean
from torch.utils.data import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    # deal with non-image datasets
    if config.data.dataset == 'Sudoku':
        dataset = Sudoku(config, is_test=False)
        test_dataset = Sudoku(config, is_test=True)
    elif config.data.dataset == 'HBMF':
        dataset = HBMF(config, is_test=False)
        test_dataset = HBMF(config, is_test=True)
    elif config.data.dataset == 'BMF':
        dataset = BMF(config, is_test=False)
        test_dataset = BMF(config, is_test=True)
    elif config.data.dataset == 'CMF':
        dataset = CMF(config, is_test=False)
        test_dataset = CMF(config, is_test=True)
    elif config.data.dataset == 'Sorting':
        dataset = Sorting(config, is_test=False)
        test_dataset = Sorting(config, is_test=True)
    elif config.data.dataset == 'Boolean':
        dataset = Boolean(config, is_test=False)
        test_dataset = Boolean(config, is_test=True)
    else:
        raise Exception
    return dataset, test_dataset

