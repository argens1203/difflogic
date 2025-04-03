import logging
import torch
from constant import device
from pysat.formula import Atom
from experiments.results_json import ResultsJSON

logger = logging.getLogger(__name__)


def feat_to_input(feat):
    inp = [idx + 1 if f == 1 else -(idx + 1) for idx, f in enumerate(feat)]
    return inp


def input_to_feat(inp):
    feat = [1 if i > 0 else 0 for i in inp]
    return torch.Tensor(feat).to(device)


def get_results(experiment_id, args):
    if experiment_id is not None:
        assert 520_000 <= experiment_id < 530_000, experiment_id
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
