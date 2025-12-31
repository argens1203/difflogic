"""Utility functions for experiment setup and data processing."""

import logging
import random
from typing import Any, Generator, Iterator, Optional

import numpy as np
import torch
from torch import Tensor
from pysat.card import EncType

from .results import Results
from constant import device


# Mapping from string encoding type names to PySAT EncType values
ENC_TYPE_MAP: dict[str, int] = {
    "pw": EncType.pairwise,
    "seqc": EncType.seqcounter,
    "sortn": EncType.sortnetwrk,
    "cardn": EncType.cardnetwrk,
    "bit": EncType.bitwise,
    "lad": EncType.ladder,
    "tot": EncType.totalizer,
    "mtot": EncType.mtotalizer,
    "kmtot": EncType.kmtotalizer,
    "native": EncType.native,
}


def seed_all(seed: int = 0) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_enc_type(enc_type: str) -> int:
    """Convert string encoding type to PySAT EncType.

    Args:
        enc_type: String name of encoding type

    Returns:
        PySAT EncType constant
    """
    return ENC_TYPE_MAP[enc_type]


def feat_to_input(feat: list[int]) -> list[int]:
    """Convert binary features to signed literal input.

    Args:
        feat: List of binary features (0 or 1)

    Returns:
        List of signed literals (positive for 1, negative for 0)
    """
    return [idx + 1 if f == 1 else -(idx + 1) for idx, f in enumerate(feat)]


def input_to_feat(inp: list[int]) -> Tensor:
    """Convert signed literal input to binary features tensor.

    Args:
        inp: List of signed literals

    Returns:
        Tensor of binary features on the configured device
    """
    feat = [1 if i > 0 else 0 for i in inp]
    return torch.Tensor(feat).to(device)


def get_results(experiment_id: int, args: Any) -> Results:
    """Create and initialize a Results object.

    Args:
        experiment_id: Unique experiment identifier
        args: Experiment arguments to store

    Returns:
        Initialized Results object
    """
    results = Results(eid=experiment_id, path="./results/")
    results.store_args(args)
    return results


def get_truth_table_loader(
    input_dim: int, batch_size: int = 10
) -> Generator[tuple[Tensor, None], None, None]:
    """Generate all possible binary inputs for a given input dimension.

    Yields batches of binary vectors covering all 2^input_dim combinations.
    Labels are set to None to match DataLoader format.

    Args:
        input_dim: Number of binary input features
        batch_size: Number of samples per batch

    Yields:
        Tuple of (input tensor, None)

    Example:
        >>> loader = get_truth_table_loader(input_dim=2, batch_size=2)
        >>> x, _ = next(loader)
        >>> print(x)
        tensor([[0, 0],
                [0, 1]])
    """
    count = 0

    def get_one(x: int) -> list[int]:
        return list(map(int, format(x, f"0{input_dim}b")))

    while count < 2**input_dim:
        batch = [get_one(i) for i in range(count, min(count + batch_size, 2**input_dim))]
        count += batch_size
        yield torch.tensor(batch), None


def remove_none(lst: list[Optional[Any]]) -> tuple[list[Any], list[int]]:
    """Remove None values from a list and track their indices.

    Args:
        lst: List potentially containing None values

    Returns:
        Tuple of (filtered list without Nones, indices of removed Nones)
    """
    ret: list[Any] = []
    indices: list[int] = []
    for idx, item in enumerate(lst):
        if item is not None:
            ret.append(item)
        else:
            indices.append(idx)
    return ret, indices


def get_onehot_loader(
    input_dim: int, attribute_ranges: list[int], batch_size: int = 10
) -> Generator[tuple[Tensor, None], None, None]:
    """Generate all possible one-hot encoded inputs for given attribute ranges.

    Each attribute range gets exactly one 1, representing a one-hot encoding.

    Args:
        input_dim: Total input dimension (should equal sum of attribute_ranges)
        attribute_ranges: List of ints representing size of each attribute group
        batch_size: Number of samples per batch

    Yields:
        Tuple of (input tensor, None)

    Example:
        >>> # For 2 attributes of size 2 and 3 respectively
        >>> loader = get_onehot_loader(input_dim=5, attribute_ranges=[2, 3], batch_size=2)
        >>> x, _ = next(loader)
        >>> print(x)
        tensor([[1, 0, 1, 0, 0],
                [1, 0, 0, 1, 0]])
    """
    assert (
        sum(attribute_ranges) == input_dim
    ), f"Sum of attribute_ranges ({sum(attribute_ranges)}) must equal input_dim ({input_dim})"

    # Calculate total number of possible combinations
    total_combinations = 1
    for range_size in attribute_ranges:
        total_combinations *= range_size

    count = 0

    def get_onehot_combination(combo_idx: int) -> list[int]:
        """Convert combination index to one-hot encoded vector."""
        result = [0] * input_dim
        offset = 0

        for range_size in attribute_ranges:
            position_in_range = combo_idx % range_size
            result[offset + position_in_range] = 1
            offset += range_size
            combo_idx //= range_size

        return result

    while count < total_combinations:
        batch = []
        for i in range(count, min(count + batch_size, total_combinations)):
            batch.append(get_onehot_combination(i))
        count += batch_size
        yield torch.tensor(batch), None
