import logging
import torch
from constant import device
from pysat.formula import Atom
from lgn.util.results_json import ResultsJSON

logger = logging.getLogger(__name__)


def feat_to_input(feat):
    inp = [idx + 1 if f == 1 else -(idx + 1) for idx, f in enumerate(feat)]
    return inp


def input_to_feat(inp):
    feat = [1 if i > 0 else 0 for i in inp]
    return torch.Tensor(feat).to(device)


def get_results(experiment_id, args):
    if experiment_id is not None:
        # assert 520_000 <= experiment_id < 530_000, experiment_id
        results = ResultsJSON(eid=experiment_id, path="./results/")
        results.store_args(args)
        return results
    return None


def get_truth_table_loader(input_dim, batch_size=10):
    """
    This generator yields all possible binary inputs for a given input dimension. The labels are set to None to adhere to the format of DataLodaers.

    returns: (x, None)

    Example:

    .. code-block:: python
        >>> instances = next(get_truth_table_loader(input_dim=2, batch_size=2))
        >>> print(instances)
        (tensor([[0, 0],
                [0, 1]]), None)
    """
    count = 0

    def get_one(x):
        return list(map(int, format(x, f"0{input_dim}b")))

    while count < 2**input_dim:
        l = [get_one(i) for i in range(count, min(count + batch_size, 2**input_dim))]
        count += batch_size
        yield torch.tensor(l), None


def remove_none(lst):
    ret = []
    indices = []
    for idx, l in enumerate(lst):
        if l is not None:
            ret.append(l)
        else:
            indices.append(idx)
    return ret, indices


def get_onehot_loader(input_dim, attribute_ranges, batch_size=10):
    """
    This generator yields all possible one-hot encoded inputs for given attribute ranges.
    Each attribute range gets exactly one 1, representing a one-hot encoding.

    Args:
        input_dim: Total input dimension (should equal sum of attribute_ranges)
        attribute_ranges: List of ints representing size of each attribute group
        batch_size: Number of samples per batch

    Returns: (x, None)

    Example:

    .. code-block:: python
        >>> # For 2 attributes of size 2 and 3 respectively
        >>> loader = get_onehot_loader(input_dim=5, attribute_ranges=[2, 3], batch_size=2)
        >>> instances = next(loader)
        >>> print(instances)
        (tensor([[1, 0, 1, 0, 0],
                [1, 0, 0, 1, 0]]), None)
    """
    assert (
        sum(attribute_ranges) == input_dim
    ), f"Sum of attribute_ranges ({sum(attribute_ranges)}) must equal input_dim ({input_dim})"

    # Calculate total number of possible combinations
    total_combinations = 1
    for range_size in attribute_ranges:
        total_combinations *= range_size

    count = 0

    def get_onehot_combination(combo_idx):
        """Convert combination index to one-hot encoded vector"""
        result = [0] * input_dim
        offset = 0

        for range_size in attribute_ranges:
            # Get which position in this attribute range should be 1
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
