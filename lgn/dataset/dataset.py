import logging
import torch

from lgn.dataset.lending import LendingDataset

from .adult import AdultDataset
from .monk import MonkDataset
from .iris import IrisDataset
from .breast_cancer import BreastCancerDataset
from .mnist import MNISTDataset
from .lending import LendingDataset
from .compas import CompasDataset


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
    if dataset == "mnist":
        return MNISTDataset.get_input_dim()
    if dataset == "lending":
        return LendingDataset.get_input_dim()
    if dataset == "compas":
        return CompasDataset.get_input_dim()
    return {
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
    if dataset == "mnist":
        return MNISTDataset.get_num_of_classes()
    if dataset == "lending":
        return LendingDataset.get_num_of_classes()
    if dataset == "compas":
        return CompasDataset.get_num_of_classes()
    return {
        "mnist20x20": 10,
        "cifar-10-3-thresholds": 10,
        "cifar-10-31-thresholds": 10,
        "caltech101": 101,
    }[dataset]


def get_dataset(dataset):
    if dataset == "adult":
        return AdultDataset
    if dataset in ["monk1", "monk2", "monk3"]:
        return MonkDataset
    if dataset == "iris":
        return IrisDataset
    if dataset == "breast_cancer":
        return BreastCancerDataset
    if dataset == "mnist":
        return MNISTDataset
    if dataset == "lending":
        return LendingDataset
    if dataset == "compas":
        return CompasDataset


class Flatten:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


import torchvision.datasets
from torchvision import transforms


class Caltech101Dataset:
    def __call__(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
