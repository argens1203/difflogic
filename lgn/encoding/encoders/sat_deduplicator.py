import logging
from pysat.formula import Formula, Atom, Or, Neg, CNF

from typing import Set

logger = logging.getLogger(__name__)


class DeduplicationMixin:
    def add_clause(self, clause: list[Formula]):
        # print("clause", clause)
        f = Or(*[x for x in clause])
        # print("f", f.simplified())
        # print("list(f)", list(f.simplified()))
        assert None not in f.simplified(), "None found in filtered clauses"
        for x in f.simplified():
            assert None not in x, "None found in filtered clauses"

        # logger.debug("Appending formula: %s", filtered)
        self.solver.append_formula(list(f.simplified()))

    def deduplicate_constant(self, f: Formula):
        # print("deduplicate_constant", f)
        with self.use_context() as vpool:
            auxvar = Atom((vpool._next()))
            auxvar_id = vpool.id(auxvar)

            self.add_clause([auxvar, f])
            self.add_clause([Neg(auxvar), Neg(f)])

            if not self.solver.solve(assumptions=[-auxvar_id]):
                # print("is constant false", f)
                self.add_clause([auxvar])
                return False

            if not self.solver.solve(assumptions=[auxvar_id]):
                # print("is constant true", f)
                self.add_clause([Neg(auxvar)])
                return True

            return None

    def deduplicate_pair(self, f: Formula, prev: Formula):
        # print("deduplicate pair", f, prev)
        with self.use_context() as vpool:
            auxvar = Atom((vpool._next()))
            auxvar_id = vpool.id(auxvar)

            self.add_clause([Neg(auxvar), Neg(f), prev])
            self.add_clause([Neg(auxvar), f, Neg(prev)])
            self.add_clause([auxvar, f, prev])
            self.add_clause([auxvar, Neg(f), Neg(prev)])

            if not self.solver.solve(assumptions=[-auxvar_id]):
                self.add_clause([auxvar])
                return True

            if not self.solver.solve(assumptions=[auxvar_id]):
                self.add_clause([Neg(auxvar)])
                return False

        return None

    def deduplicate(self, f: Formula, previous: Set[Formula]):
        c = self.deduplicate_constant(f)
        if c is not None:
            self.e_ctx.inc_deduplication()
            return Atom(c)

        for p in previous:
            if len(str(f)) <= len(str(p)):
                continue
            g = self.deduplicate_pair(f, p)
            if g is not None:
                self.e_ctx.inc_deduplication()
                if g is True:
                    return p
                else:
                    return Neg(p)
        assert "None" not in str(f), "Deduplication returned None for formula"
        return f
