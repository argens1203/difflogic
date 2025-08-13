import unittest
import argparse

import pytest

from constant import Stats

from lgn.dataset.iris import IrisDataset
from lgn.dataset.loader import load_dataset
from lgn.encoding import Encoder

from experiment.helpers import get_results
from experiment.args import DefaultArgs
from experiment.settings import Settings

from lgn.model.model import get_model

default_args = DefaultArgs()


class TestSat(unittest.TestCase):
    def setup_method(self, method):
        dataset = "iris"
        dataset_args: dict[str, int] = Settings.debug_network_param.get(dataset) or {}
        exp_args = {
            "eval_freq": 1000,
            "model_path": dataset + "_" + "model.pth",
            "verbose": "info",
            "save_model": True,
            "load_model": True,
        }
        args = {
            **vars(default_args),
            **exp_args,
            **dataset_args,
            **{"dataset": dataset},
        }
        args = argparse.Namespace(**args)
        _, __, ___, dataset = load_dataset(args)
        model, loss_fn, optim = get_model(args, get_results(0, args))

        self.encoding = Encoder().get_encoding(model, IrisDataset())
        Stats["deduplication"] = 0

    @pytest.mark.skip("cannot deduplicate without an encoding")
    def test_xxx(self):
        pass
        # ctx = Context(args)
        # model = Model.get_model(args, ctx=ctx)
        # sswd = SatEncoder(model=model)

        # res = sswd.deduplicate_constant(Atom(True))
        # print("deduplication result", res)
        # assert res == True, "Expected True for constant True"

        # res = sswd.deduplicate_constant(Atom(False))
        # print("deduplication result", res)
        # assert res == False, "Expected False for constant False"

        # res = sswd.deduplicate_constant(Atom("lksjflsdkfj"))
        # print("deduplication result", res)
