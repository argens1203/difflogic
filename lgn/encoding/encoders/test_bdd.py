import pytest

from experiment.helpers.context import Context
from experiment.args import DefaultArgs

from .bdd_deduplicator import BDDSolver, xor

from pysat.formula import Atom, Or, XOr, And, Implies, Neg


@pytest.fixture
def e_ctx():
    """Create a context for testing."""
    return Context(args=DefaultArgs())


@pytest.fixture
def solver(e_ctx):
    """Create a BDD solver with 4 variables."""
    return BDDSolver([1, 2, 3, 4], e_ctx=e_ctx)


@pytest.fixture
def solver_with_ohe(e_ctx):
    """Create a BDD solver with OHE constraints [2, 2]."""
    s = BDDSolver([1, 2, 3, 4], e_ctx=e_ctx)
    s.set_ohe([2, 2])
    return s


class TestBddSolver:
    def test_solver_equiv(self, solver):
        bdd = solver.bdd

        u = ~bdd.var(1) | ~bdd.var(2)
        v = ~(bdd.var(2) & bdd.var(1))

        assert solver.is_equiv(u, v) is True

    def test_ohe_positive_cases(self, solver_with_ohe):
        solver = solver_with_ohe
        bdd = solver.bdd

        u = xor(bdd.var(1), bdd.var(2))
        v = bdd.var(1) | bdd.var(2)
        assert solver.is_equiv(u, v) is True

        a = bdd.false
        b = bdd.var(3) & bdd.var(4)
        assert solver.is_equiv(a, b) is True

    def test_ohe_atomic_cases(self, solver_with_ohe):
        solver = solver_with_ohe
        bdd = solver.bdd

        assert solver.is_equiv(~(bdd.var(3).implies(bdd.var(4))), bdd.var(3)) is True
        assert (
            solver.is_neg_equiv(~(bdd.var(3).implies(bdd.var(4))), ~bdd.var(3)) is True
        )

        assert solver.is_equiv(~(bdd.var(3).implies(bdd.var(4))), ~bdd.var(4)) is True
        assert (
            solver.is_neg_equiv(~(bdd.var(3).implies(bdd.var(4))), bdd.var(4)) is True
        )

    def test_ohe_implied(self, solver_with_ohe):
        solver = solver_with_ohe
        bdd = solver.bdd

        assert solver.is_equiv(bdd.var(1), ~bdd.var(2)) is True
        assert solver.is_equiv(~bdd.var(1), bdd.var(2)) is True
        assert solver.is_neg_equiv(bdd.var(1), bdd.var(2)) is True

        assert solver.is_equiv(bdd.var(3), ~bdd.var(4)) is True
        assert solver.is_equiv(~bdd.var(3), bdd.var(4)) is True
        assert solver.is_neg_equiv(bdd.var(3), bdd.var(4)) is True

    def test_ohe_not_overly_restrictive(self, solver_with_ohe):
        solver = solver_with_ohe
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

    def test_ohe_with_more_features(self, e_ctx):
        solver = BDDSolver([1, 2, 3, 4, 5], e_ctx=e_ctx)
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

    def test_deduplicate(self, solver_with_ohe):
        solver = solver_with_ohe

        f1 = XOr(Atom(1), Atom(2))
        f2 = Or(Atom(1), Atom(2))

        bdd = solver.bdd

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

    def test_reduced_variable_domain(self, e_ctx):
        solver = BDDSolver(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            e_ctx=e_ctx,
        )
        solver.set_ohe([3, 3, 2, 3, 4, 2])

        f1 = Atom(5)
        f2 = Neg(Implies(Atom(5), Atom(4)))

        assert solver.is_equiv(solver.transform(f1), solver.transform(f2)) is True
