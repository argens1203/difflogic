import logging
import torch
import torchvision

from abc import ABC, abstractmethod

from .binarizer import Binarizer

logger = logging.getLogger(__name__)


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def input_dim_of_dataset(dataset):  # TODO: get it from Dataset class
    return {
        "adult": 115,
        "breast_cancer": 51,
        "iris": 4 * 2,
        "monk1": 17,
        "monk2": 17,
        "monk3": 17,
        "mnist": 400 * 2,
        "mnist20x20": 400,
        "cifar-10-3-thresholds": 3 * 32 * 32 * 3,
        "cifar-10-31-thresholds": 3 * 32 * 32 * 31,
        "caltech101": 64 * 64 * 2,
    }[dataset]


def num_classes_of_dataset(dataset):  # TODO: get it from Dataset class
    return {
        "adult": 2,
        "breast_cancer": 2,
        "iris": 3,
        "monk1": 2,
        "monk2": 2,
        "monk3": 2,
        "mnist": 10,
        "mnist20x20": 10,
        "cifar-10-3-thresholds": 10,
        "cifar-10-31-thresholds": 10,
        "caltech101": 101,
    }[dataset]


def get_attribute_ranges(dataset):
    return {
        "monk1": [3, 3, 2, 3, 4, 2],
        "monk2": [3, 3, 2, 3, 4, 2],
        "monk3": [3, 3, 2, 3, 4, 2],
        "iris": [2, 2, 2, 2],
        "breast_cancer": [9, 3, 12, 13, 2, 3, 2, 5, 2],
        "adult": [5, 7, 16, 7, 14, 6, 5, 2, 3, 3, 6, 41],
    }[dataset]


from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset, DataLoader
import os

import numpy as np


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
        return feature, label

    def get_all(self):
        return self.features, self.labels

    def read_raw_data(self, filepath, delimiter=",", select=lambda x: True):
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


class IrisDataset(CustomDataset):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "42615765a885ddf54427f12c34a0a070",
    )
    # location = "iris.data"

    label_dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    def load_data(self):
        def parse_feature(features):
            return [float(f) for f in features]

        def parse(data):
            features = [parse_feature(sample[:-1]) for sample in data]
            labels = [self.label_dict[sample[-1]] for sample in data]
            features, labels = (
                torch.tensor(features).float(),
                torch.tensor(labels).float(),
            )

            return features, labels

        raw_data = self.read_raw_data(self.fpath)
        self.features, self.labels = parse(raw_data)


class AdultDataset(CustomDataset):
    file_list = [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "5d7c39d7b8804f071cdd1f2a7c460872",
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
            "35238206dfdf7f1fe215bbb874adecdc",
        ),
    ]
    label_dict = {"<=50K": 0, ">50K": 1}

    def __init__(self, train=True, transform=None):
        self.url, self.md5 = self.file_list[0] if train else self.file_list[1]
        CustomDataset.__init__(self, transform)

    def load_data(self):
        def parse_feature(features):
            return [float(f) for f in features]

        def parse(data):
            features = [parse_feature(sample[:-1]) for sample in data]
            labels = [self.label_dict[sample[-1]] for sample in data]
            features, labels = (
                torch.tensor(features).float(),
                torch.tensor(labels).float(),
            )

            return features, labels

        raw_data = self.read_raw_data(self.fpath)
        self.features, self.labels = parse(raw_data)


class BreastCancerDataset(CustomDataset):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data",
        "d887db31e2b99e39c4b7fc79d2f36a3a",
    )

    def __init__(self, train=True, transform=None):
        CustomDataset.__init__(self, transform)


import torch.nn.functional as F


class Flatten:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


import torchvision.datasets
from torchvision import transforms


class Caltech101Dataset:
    dataset = torchvision.datasets.Caltech101(
        "data-uci",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.Grayscale(),
                Flatten(),
            ]
        ),
    )
    dataset = torchvision.datasets.Caltech101(
        "data-uci",
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.Grayscale(),
                Flatten(),
                Binarizer(dataset, 2),
            ]
        ),
    )

    def __call__(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class MNISTDataset:
    dataset = torchvision.datasets.MNIST(
        "data-uci",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((20, 20)),
                Flatten(),
            ]
        ),
    )
    dataset = torchvision.datasets.MNIST(
        "data-uci",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((20, 20)),
                Flatten(),
                Binarizer(dataset, 2),
            ]
        ),
    )

    def __call__(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
