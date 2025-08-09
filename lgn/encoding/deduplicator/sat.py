from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from pysat.formula import Formula, Atom, CNF, Or, Neg, PYSAT_TRUE, PYSAT_FALSE, Implies

from typing import Set

from lgn.explanation.solver import Solver

if TYPE_CHECKING:
    from lgn.encoding import Encoding
from constant import Stats

logger = logging.getLogger(__name__)


class DeduplicationMixin:
    def add_clause(self, clause: list[Formula]):
        # print("Adding clause: ", clause)
        # logger.debug(clause)
        if Atom(True) in [x.simplified() for x in clause]:
            # print("Clause contains True, skipping")
            return

        f = Or(*[x.simplified() for x in clause])
        f.clausify()
        # logger.debug("f.clauses", f.clauses)

        clauses = list(filter(lambda x: x is not None, f.clauses))
        clauses = list(map(lambda x: list(filter(lambda y: y is not None, x)), clauses))
        filtered = list(filter(lambda z: z is not None and len(z) > 0, clauses))
        assert None not in filtered, "None found in filtered clauses"

        # logger.debug("Appending formula: %s", filtered)
        self.solver.append_formula(filtered)

    def deduplicate_constant(self, f: Formula):
        with self.use_context() as vpool:
            # print("------- deduplicating constant ------", f)
            auxvar = Atom(("constant", f))
            auxvar_id = vpool.id(auxvar)
            # print("Added auxvar to vpool", auxvar, auxvar_id)

            self.add_clause([auxvar, f])
            self.add_clause([Neg(auxvar), Neg(f)])

            if not self.solver.solve(assumptions=[-auxvar_id]):
                # print("is constant false", f)
                self.add_clause([auxvar])
                Stats["deduplication"] += 1
                return False

            if not self.solver.solve(assumptions=[auxvar_id]):
                # print("is constant true", f)
                self.add_clause([Neg(auxvar)])
                Stats["deduplication"] += 1
                return True

            return None

    def deduplicate_pair(self, f: Formula, prev: Formula):
        with self.use_context() as vpool:
            if len(str(f)) <= len(str(prev)):
                return None

            # print("------- deduplicating pair ------", f, prev)

            auxvar = Atom(("pair", f, prev))
            auxvar_id = vpool.id(auxvar)
            # self.add_clause([auxvar])
            # self.add_clause([Neg(auxvar)])

            self.add_clause([Neg(auxvar), Neg(f), prev])
            self.add_clause([Neg(auxvar), f, Neg(prev)])
            self.add_clause([auxvar, f, prev])
            self.add_clause([auxvar, Neg(f), Neg(prev)])

            if not self.solver.solve(assumptions=[-auxvar_id]):
                self.add_clause([auxvar])
                Stats["deduplication"] += 1
                return True

            if not self.solver.solve(assumptions=[auxvar_id]):
                self.add_clause([Neg(auxvar)])
                Stats["deduplication"] += 1
                return False

        return None

    def deduplicate(self, f: Formula, previous: Set[Formula]):
        c = self.deduplicate_constant(f)
        if c is not None:
            # logger.debug(f"Deduplicated {f} to {c}")
            return Atom(c)

        for p in previous:
            g = self.deduplicate_pair(f, p)
            if g is not None:
                # logger.debug(f"Deduplicated {f} with {p} to {g}")
                if g is True:
                    return p
                else:
                    return Neg(p)
        # logger.debug(str(f))
        assert "None" not in str(f), "Deduplication returned None for formula"
        return f
