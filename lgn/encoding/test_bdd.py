import unittest
from .bdd import BDDSolver, xor

from pysat.formula import Formula, Atom, CNF, Or, XOr, And


class TestBdd(unittest.TestCase):
    def test_solver_equiv(self):
        solver = BDDSolver([1, 2, 3, 4])
        bdd = solver.bdd

        u = ~bdd.var(1) | ~bdd.var(2)
        v = ~(bdd.var(2) & bdd.var(1))

        assert solver.is_equiv(u, v) is True
        assert solver.is_equiv2(u, v) is True

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

        res = solver.deduplicate([f1, f2])
        assert res[0] == res[1], (res[0], res[1])

        f1 = Atom(False)
        f2 = And(Atom(3), Atom(4))
        res = solver.deduplicate([f1, f2])
        assert res[0] == res[1], (res[0], res[1])
