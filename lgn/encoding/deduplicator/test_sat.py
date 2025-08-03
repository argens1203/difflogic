import unittest
import argparse

import pytest

from constant import Stats
from lgn.dataset.iris import IrisDataset
from lgn.dataset.loader import new_load_dataset
from lgn.encoding import Encoder
from lgn.deduplicator import SatDeduplicator
from lgn.util.util import get_results
from .bdd import BDDSolver, xor

from pysat.formula import Atom, Or, XOr, And
from lgn.encoding import Encoding
from lgn.model.model import get_model
from lgn.experiment.settings import Settings
from pysat.formula import PYSAT_FALSE, PYSAT_TRUE
from lgn.util import DefaultArgs

default_args = DefaultArgs()


class TestSat(unittest.TestCase):
    def setup_method(self, method):
        dataset = "iris"
        dataset_args: dict[str, int] = Settings.debug_network_param.get(dataset) or {}
        exp_args = {
            "eval_freq": 1000,
            "model_path": dataset + "_" + "model.pth",
            "verbose": True,
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
        _, __, ___, dataset = new_load_dataset(args)
        model, loss_fn, optim = get_model(args, get_results(0, args))

        self.encoding = Encoder().get_static(model, IrisDataset())
        Stats["deduplication"] = 0

    def test_xxx(self):
        sswd = SatDeduplicator(self.encoding)

        res = sswd.deduplicate_constant(Atom(True))
        print("deduplication result", res)
        assert res == True, "Expected True for constant True"

        res = sswd.deduplicate_constant(Atom(False))
        print("deduplication result", res)
        assert res == False, "Expected False for constant False"

        res = sswd.deduplicate_constant(Atom("lksjflsdkfj"))
        print("deduplication result", res)
        assert res is None, "Expected None for Variable 1"

    @pytest.mark.skip("wip")
    def test_solver_equiv(self):
        solver = BDDSolver([1, 2, 3, 4])
        bdd = solver.bdd

        u = ~bdd.var(1) | ~bdd.var(2)
        v = ~(bdd.var(2) & bdd.var(1))

        assert solver.is_equiv(u, v) is True
        assert solver.is_equiv2(u, v) is True

    @pytest.mark.skip("wip")
    def test_ohe_positive_cases(self):
        solver = BDDSolver([1, 2, 3, 4])
        solver.set_ohe([2, 2])

        bdd = solver.bdd

        u = xor(bdd.var(1), bdd.var(2))
        v = bdd.var(1) | bdd.var(2)
        assert solver.is_equiv(u, v) is True

        a = bdd.false
        b = bdd.var(3) & bdd.var(4)
        assert solver.is_equiv(a, b) is True

    @pytest.mark.skip("wip")
    def test_ohe_not_overly_restrictive(self):
        solver = BDDSolver([1, 2, 3, 4])
        solver.set_ohe([2, 2])

        bdd = solver.bdd

        u = xor(bdd.var(2), bdd.var(3))
        v = bdd.var(2) | bdd.var(3)
        assert solver.is_equiv(u, v) is False

        a = bdd.false
        b = bdd.var(1) & bdd.var(4)
        assert solver.is_equiv(a, b) is False

        x = bdd.var(1) | bdd.var(2) | bdd.var(3)
        y = xor(bdd.var(1), bdd.var(2)) | bdd.var(3)
        assert solver.is_equiv(x, y) is True

    @pytest.mark.skip("wip")
    def test_ohe_with_more_features(self):
        solver = BDDSolver([1, 2, 3, 4, 5])
        solver.set_ohe([5])

        bdd = solver.bdd

        p = bdd.var(1) & bdd.var(2)
        assert solver.is_equiv(p, bdd.false) is True

        p = ~bdd.var(1) & ~bdd.var(2) & ~bdd.var(3) & ~bdd.var(4) & ~bdd.var(5)
        assert solver.is_equiv(p, bdd.false) is True

        p = bdd.var(1) & ~bdd.var(2) & ~bdd.var(3) & ~bdd.var(4) & ~bdd.var(5)
        assert solver.is_equiv(p, bdd.false) is False

        p = ~bdd.var(2) & ~bdd.var(3) & ~bdd.var(4) & ~bdd.var(5)
        assert solver.is_equiv(p, bdd.false) is False

    @pytest.mark.skip("wip")
    def test_deduplicate(self):
        solver = BDDSolver([1, 2, 3, 4])
        solver.set_ohe([2, 2])

        f1 = XOr(Atom(1), Atom(2))
        f2 = Or(Atom(1), Atom(2))

        bdd = solver.bdd
        print(bdd.var(1), bdd.var(2))
        print(bdd.var(1))

        assert solver.is_equiv(solver.transform(Atom(1)), bdd.var(1)) is True

        assert (
            solver.is_equiv(solver.transform(f1), xor(bdd.var(1), bdd.var(2))) is True
        )
        bdd = solver.bdd
        assert solver.is_equiv(solver.transform(f2), bdd.var(1) | bdd.var(2)) is True

        assert (solver.is_equiv(solver.transform(f1), solver.transform(f2))) is True

        res1 = solver.deduplicate(f2, set())
        res2 = solver.deduplicate(f1, set([f2]))
        assert res1 == res2, (res1, res2)

        f1 = Atom(False)
        f2 = And(Atom(3), Atom(4))
        res = solver.deduplicate(f1, set([f2]))
        assert f1 == res, (f1, res)
