from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset
import os

import numpy as np

from abc import ABC, abstractmethod


class CustomDataset(Dataset, ABC):
    root = "data-uci"

    def __init__(self, transform=None, root=None):
        self.root = root if root is not None else self.root
        self.transform = transform
        self.fpath = os.path.join(self.root, self.url.split("/")[-1])

        if not check_integrity(self.fpath, self.md5):
            download_url(self.url, self.root, self.url.split("/")[-1], self.md5)
        self.load_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]

        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label, index

    def get_all(self):
        return self.features, self.labels

    def read_raw_data(self, filepath=None, delimiter=",", select=lambda x: True):
        if filepath is None:
            filepath = self.fpath

        with open(filepath, "r") as f:
            data = f.readlines()

        for i in range(len(data)):
            if len(data[i]) <= 2 or not select(data[i]):
                data[i] = None
            else:
                data[i] = data[i].strip("\n").strip().strip(".").split(delimiter)
                data[i] = [d.strip() for d in data[i]]
        data = list(filter(lambda x: x is not None, data))
        return np.array(data)

    @abstractmethod
    def load_data(self):
        pass

    def _get_fpath(self):
        return self.fpath
