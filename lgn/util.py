import torch
from constant import device
from pysat.formula import Atom
from experiments.results_json import ResultsJSON


def get_results(experiment_id, args):
    if experiment_id is not None:
        assert 520_000 <= experiment_id < 530_000, experiment_id
        results = ResultsJSON(eid=experiment_id, path="./results/")
        results.store_args(args)
        return results
    return None


def summed_on(x, class_count_to_conform_to):
    # TODO: use torch way
    assert len(x) % class_count_to_conform_to == 0
    ret = []
    for i in range(0, len(x), len(x) // class_count_to_conform_to):
        ret.append(sum(x[i : i + len(x) // class_count_to_conform_to]))
    return ret


def formula_as_pseudo_model(formula, input_handles, class_dim):
    def pseudo_model(x):
        simplified = [
            [
                (
                    0
                    if f.simplified(
                        assumptions=[
                            ~inp if feat == 0 else inp
                            for feat, inp in zip(xs, input_handles)
                        ]
                    )
                    # TODO: better way of checking?
                    == Atom(False)
                    else 1
                )
                for f in formula
            ]
            for xs in x
        ]
        summed = (
            torch.tensor([summed_on(inter, class_dim) for inter in simplified])
            .to(device)
            .int()
        )
        return summed

    return pseudo_model


def get_truth_table_loader(input_dim, batch_size=10):
    count = 0

    def get_one(x):
        return list(map(int, format(x, f"0{input_dim}b")))

    while count < 2**input_dim:
        l = [get_one(i) for i in range(count, min(count + batch_size, 2**input_dim))]
        count += batch_size
        yield torch.tensor(l), None
