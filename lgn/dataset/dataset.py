import logging
import torch

from .adult import AdultDataset
from .monk import MonkDataset
from .iris import IrisDataset
from .breast_cancer import BreastCancerDataset
from .mnist import MNISTDataset
from .lending import LendingDataset
from .compas import CompasDataset


# Dataset registry mapping names to dataset classes
# Each dataset class must implement get_input_dim() and get_num_of_classes()
DATASET_REGISTRY = {
    "iris": IrisDataset,
    "adult": AdultDataset,
    "monk1": MonkDataset,
    "monk2": MonkDataset,
    "monk3": MonkDataset,
    "breast_cancer": BreastCancerDataset,
    "mnist": MNISTDataset,
    "lending": LendingDataset,
    "compas": CompasDataset,
}

# Legacy datasets that don't have proper dataset classes yet
# These are image datasets with hardcoded dimensions
LEGACY_DATASETS = {
    "mnist20x20": {"input_dim": 400, "num_classes": 10},
    "cifar-10-3-thresholds": {"input_dim": 3 * 32 * 32 * 3, "num_classes": 10},
    "cifar-10-31-thresholds": {"input_dim": 3 * 32 * 32 * 31, "num_classes": 10},
    "caltech101": {"input_dim": 64 * 64 * 2, "num_classes": 101},
}


def get_dataset(dataset_name: str):
    """
    Get the dataset class for the given dataset name.

    Args:
        dataset_name: Name of the dataset (e.g., 'iris', 'mnist', 'monk1')

    Returns:
        The dataset class (not an instance)

    Raises:
        KeyError: If the dataset is not found in the registry
    """
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name]
    raise KeyError(f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}")


def input_dim_of_dataset(dataset_name: str) -> int:
    """
    Get the input dimension for the given dataset.

    Note: For registry datasets, this requires the dataset class to have been
    instantiated at least once (to set up the converter). For legacy datasets,
    returns hardcoded values.

    Args:
        dataset_name: Name of the dataset

    Returns:
        The input dimension (number of features after binarization)
    """
    if dataset_name in LEGACY_DATASETS:
        return LEGACY_DATASETS[dataset_name]["input_dim"]
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name].get_input_dim()
    raise KeyError(f"Dataset '{dataset_name}' not found")


def num_classes_of_dataset(dataset_name: str) -> int:
    """
    Get the number of classes for the given dataset.

    Note: For registry datasets, this requires the dataset class to have been
    instantiated at least once (to set up the label encoder). For legacy datasets,
    returns hardcoded values.

    Args:
        dataset_name: Name of the dataset

    Returns:
        The number of output classes
    """
    if dataset_name in LEGACY_DATASETS:
        return LEGACY_DATASETS[dataset_name]["num_classes"]
    if dataset_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_name].get_num_of_classes()
    raise KeyError(f"Dataset '{dataset_name}' not found")


def load_n(loader, n):
    """Load n samples from a data loader."""
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


class Flatten:
    """Transform to flatten a tensor to 1D."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


class Caltech101Dataset:
    """Placeholder for Caltech101 dataset (not fully implemented)."""
    def __call__(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
