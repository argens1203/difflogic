from __future__ import annotations
from typing import TYPE_CHECKING
from pysat.formula import Formula, Atom, CNF, Or, Neg, PYSAT_TRUE, PYSAT_FALSE

from typing import Set

from lgn.explanation.solver import Solver

if TYPE_CHECKING:
    from lgn.encoding import Encoding
from constant import Stats


class SolverWithDeduplication(Solver):
    def __init__(self, encoding: Encoding):
        super().__init__(encoding)

    def add_clause(self, clause: list[Formula]):
        # self.solver.add_clause(clause)
        for f in [Or(Atom(False), f.simplified()) for f in clause]:
            cnf = CNF()
            f.clausify()
            cnf.extend(list(f)[:-1])

        self.solver.append_formula(cnf)

    def deduplicate_constant(self, f: Formula):
        # TODO: test minus sign
        # TODO: use variable to store the output
        with self.use_context() as vpool:
            auxvar = Atom(("constant", f))
            auxvar_id = vpool.id(auxvar)
            self.add_clause([auxvar, f])
            self.add_clause([Neg(auxvar), Neg(f)])

            if not self.solver.solve(assumptions=[-auxvar_id]):
                self.add_clause([auxvar])
                Stats["deduplication"] += 1
                return PYSAT_TRUE

            if not self.solver.solve(assumptions=[auxvar_id]):
                self.add_clause([Neg(auxvar)])
                Stats["deduplication"] += 1
                return PYSAT_FALSE

            return None

    def deduplicate_pair(self, f: Formula, prev: Formula):
        with self.use_context() as vpool:
            auxvar = Atom(("pair", f, prev))
            auxvar_id = vpool.id(auxvar)
            self.add_clause([Neg(auxvar), Neg(f), prev])
            self.add_clause([Neg(auxvar), f, Neg(prev)])
            self.add_clause([auxvar, f, prev])
            self.add_clause([auxvar, Neg(f), Neg(prev)])
            if not self.solver.solve(assumptions=[-auxvar_id]):
                self.add_clause([auxvar])
                Stats["deduplication"] += 1
                return prev

            if not self.solver.solve(assumptions=[auxvar_id]):
                self.add_clause([Neg(auxvar)])
                Stats["deduplication"] += 1
                return Neg(prev)

        return None

    def deduplicate(self, f: Formula, previous: Set[Formula]):
        c = self.deduplicate_constant(f)
        if c is not None:
            return c

        for p in previous:
            g = self.deduplicate_pair(f, p)
            if g is not None:
                return g

        return f
