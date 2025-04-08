import logging
import torch
import torchvision


from .binarizer import Binarizer
from .adult import AdultDataset
from .monk import MonkDataset
from .iris import IrisDataset
from .breast_cancer import BreastCancerDataset

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
    if dataset in ["monk1", "monk2", "monk3"]:
        return MonkDataset.get_input_dim()
    if dataset == "iris":
        return IrisDataset.get_input_dim()
    if dataset == "breast_cancer":
        return BreastCancerDataset.get_input_dim()
    return {
        "mnist": 400 * 2,
        "mnist20x20": 400,
        "cifar-10-3-thresholds": 3 * 32 * 32 * 3,
        "cifar-10-31-thresholds": 3 * 32 * 32 * 31,
        "caltech101": 64 * 64 * 2,
    }[dataset]


def num_classes_of_dataset(dataset):  # TODO: get it from Dataset class
    if dataset == "adult":
        return AdultDataset.get_num_of_classes()
    if dataset in ["monk1", "monk2", "monk3"]:
        return MonkDataset.get_num_of_classes()
    if dataset == "iris":
        return IrisDataset.get_num_of_classes()
    if dataset == "breast_cancer":
        return BreastCancerDataset.get_num_of_classes()
    return {
        "mnist": 10,
        "mnist20x20": 10,
        "cifar-10-3-thresholds": 10,
        "cifar-10-31-thresholds": 10,
        "caltech101": 101,
    }[dataset]


def get_attribute_ranges(dataset):
    if dataset == "adult":
        return AdultDataset
    if dataset in ["monk1", "monk2", "monk3"]:
        return MonkDataset
    if dataset == "iris":
        return IrisDataset
    if dataset == "breast_cancer":
        return BreastCancerDataset


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
