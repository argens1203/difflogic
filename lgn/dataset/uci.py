import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity
import os
from sklearn.model_selection import train_test_split


class UCIDataset(Dataset):
    def __init__(self, root, split="train", download=False):
        super(UCIDataset, self).__init__()
        self.root = root
        self.split = split

        if download and not self._check_integrity():
            self.downloads()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        for file in self.file_list:
            md5 = file[1]
            fpath = os.path.join(self.root, file[0].split("/")[-1])
            if not check_integrity(fpath, md5):
                return False
        return True

    def downloads(self):
        for file in self.file_list:
            md5 = file[1]
            download_url(file[0], self.root, file[0].split("/")[-1], md5)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)
