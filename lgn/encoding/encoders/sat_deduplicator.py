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
        if f == Atom(True) or f == Atom(False):
            return f

        c = self.deduplicate_constant(f)
        if c is not None:
            self.e_ctx.inc_deduplication()
            return Atom(c)

        for p in previous:
            # if len(str(f)) <= len(str(p)):
            # continue
            g = self.deduplicate_pair(f, p)
            if g is not None:
                self.e_ctx.inc_deduplication()
                if g is True:
                    return p
                else:
                    return Neg(p)
        assert "None" not in str(f), "Deduplication returned None for formula"
        return f

    def dedup_pair_c(self, gate, prev, solver):
        with self.use_context() as vpool:
            auxvar_id = vpool._next()

            solver.append_formula([[auxvar_id, -gate, prev]])
            solver.append_formula([[auxvar_id, gate, -prev]])
            solver.append_formula([[-auxvar_id, -gate, -prev]])
            solver.append_formula([[-auxvar_id, gate, prev]])

            if not solver.solve(assumptions=[-auxvar_id]):
                return [auxvar_id], True
            if not solver.solve(assumptions=[auxvar_id]):
                return [-auxvar_id], False

    def deduplicate_c(self, i, j, gates, solver, input_ids):
        # print("gates", gates)
        gate = gates[i][j]
        if not solver.solve(assumptions=[-gate]):
            # print(solver.get_core())
            # print("is constant True", gate)
            self.e_ctx.inc_deduplication()
            return [gate], None, None, True, None
        if not solver.solve(assumptions=[gate]):
            # print(solver.get_core())
            # print("is constant False", gate)
            self.e_ctx.inc_deduplication()
            return [-gate], None, None, False, None

        for k, layer in enumerate(gates):
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    return None, None, None, None, None
                res = self.dedup_pair_c(gate, prev, solver)
                if res is not None:
                    # print("deduplicate pair:", gate, prev, k, m)
                    self.e_ctx.inc_deduplication()
                    clause, is_reverse = res
                    return clause, k, m, None, is_reverse

        assert False

        #     self.add_clause([Neg(auxvar), gates[i][j]])
        #     self.add_clause([auxvar, Neg(gates[i][j])])

        #     if not solver.solve(assumptions=[-auxvar_id]):
        #         self.add_clause([auxvar])
        #         return None, i, j, False, None

        #     if not solver.solve(assumptions=[auxvar_id]):
        #         self.add_clause([Neg(auxvar)])
        #         return None, i, j, True, None

        # return None, i, j, None, None
