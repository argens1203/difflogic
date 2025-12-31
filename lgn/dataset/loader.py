from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import Caltech101Dataset, DATASET_REGISTRY
from .mnist import MNISTDatasetFactory
from .adult import AdultDataset
from .monk import Monk1Dataset, Monk2Dataset, Monk3Dataset
from .lending import LendingDataset

# Datasets that have explicit train/test splits (via split parameter)
SPLIT_DATASETS: dict[str, type] = {
    "adult": AdultDataset,
    "lending": LendingDataset,
    "monk1": Monk1Dataset,
    "monk2": Monk2Dataset,
    "monk3": Monk3Dataset,
}

# Datasets that require random splitting
RANDOM_SPLIT_DATASETS: set[str] = {"iris", "breast_cancer", "compas", "caltech101"}

# Special dataset that needs factory function
FACTORY_DATASETS: dict[str, Callable] = {
    "mnist": MNISTDatasetFactory,
}

# Default train/test split ratio for random split datasets
DEFAULT_TRAIN_RATIO = 0.8


def get_raw(
    raw: Optional[Any], train: Optional[Any], test: Optional[Any]
) -> Callable[[int, bool], Any]:
    """
    Create a function to retrieve raw features by index.

    Args:
        raw: Single dataset (for random split datasets)
        train: Training dataset (for split datasets)
        test: Test dataset (for split datasets)

    Returns:
        Function that returns raw features given index and is_train flag
    """
    def get_raw_data(index: int, is_train: bool) -> Any:
        if raw is not None:
            return raw.raw_features[index]
        if is_train:
            return train.raw_features[index]
        return test.raw_features[index]

    return get_raw_data


def _create_data_loaders(
    train_set: Any, test_set: Any, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """Create train and test data loaders."""
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _load_split_dataset(
    dataset_class: type, batch_size: int
) -> tuple[DataLoader, DataLoader, Callable, Any]:
    """Load a dataset that has explicit train/test splits."""
    train_set = dataset_class(split="train")
    test_set = dataset_class(split="test")
    train_loader, test_loader = _create_data_loaders(train_set, test_set, batch_size)
    return train_loader, test_loader, get_raw(None, train_set, test_set), train_set


def _load_random_split_dataset(
    dataset: Any, batch_size: int, train_ratio: float = DEFAULT_TRAIN_RATIO
) -> tuple[DataLoader, DataLoader, Callable, Any]:
    """Load a dataset and randomly split into train/test."""
    train_set, test_set = random_split(dataset, [train_ratio, 1 - train_ratio])
    train_loader, test_loader = _create_data_loaders(train_set, test_set, batch_size)
    return train_loader, test_loader, get_raw(dataset, None, None), dataset


def load_dataset(
    args: Any,
) -> tuple[DataLoader, DataLoader, Callable[[int, bool], Any], Any]:
    """
    Load a dataset based on the args configuration.

    Args:
        args: Configuration object with 'dataset' and 'batch_size' attributes

    Returns:
        Tuple of (train_loader, test_loader, get_raw_fn, dataset)
    """
    dataset_name = args.dataset
    batch_size = args.batch_size

    # Handle datasets with explicit train/test splits
    if dataset_name in SPLIT_DATASETS:
        return _load_split_dataset(SPLIT_DATASETS[dataset_name], batch_size)

    # Handle factory-created datasets (e.g., MNIST with custom args)
    if dataset_name in FACTORY_DATASETS:
        dataset = FACTORY_DATASETS[dataset_name](args)
        return _load_random_split_dataset(dataset, batch_size)

    # Handle datasets from the registry that use random splits
    if dataset_name in RANDOM_SPLIT_DATASETS:
        if dataset_name in DATASET_REGISTRY:
            dataset = DATASET_REGISTRY[dataset_name]()
        elif dataset_name == "caltech101":
            dataset = Caltech101Dataset()
        else:
            raise KeyError(f"Dataset '{dataset_name}' not found in registry")
        return _load_random_split_dataset(dataset, batch_size)

    # Fallback: try to get from registry
    if dataset_name in DATASET_REGISTRY:
        dataset = DATASET_REGISTRY[dataset_name]()
        return _load_random_split_dataset(dataset, batch_size)

    raise KeyError(
        f"Unknown dataset: '{dataset_name}'. "
        f"Available: {list(SPLIT_DATASETS.keys()) + list(RANDOM_SPLIT_DATASETS) + list(FACTORY_DATASETS.keys())}"
    )
