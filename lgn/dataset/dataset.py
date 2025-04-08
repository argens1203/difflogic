import logging
import torch
import torchvision


from .binarizer import Binarizer
from .adult import AdultDataset

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
    if dataset == "adult":
        return AdultDataset.get_input_dim()
    return {
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
    if dataset == "adult":
        return AdultDataset.get_attribute_ranges()
    return {
        "monk1": [3, 3, 2, 3, 4, 2],
        "monk2": [3, 3, 2, 3, 4, 2],
        "monk3": [3, 3, 2, 3, 4, 2],
        "iris": [2, 2, 2, 2],
        "breast_cancer": [9, 3, 12, 13, 2, 3, 2, 5, 2],
    }[dataset]


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
