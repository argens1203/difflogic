import logging
from pysat.formula import Formula, Atom, Or, Neg, CNF

from typing import Set

logger = logging.getLogger(__name__)


class DeduplicationMixin:
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
            return [gate], None, None, True, None
        if not solver.solve(assumptions=[gate]):
            # print(solver.get_core())
            # print("is constant False", gate)
            return [-gate], None, None, False, None

        for k, layer in enumerate(gates):
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    return None, None, None, None, None
                res = self.dedup_pair_c(gate, prev, solver)
                if res is not None:
                    # print("deduplicate pair:", gate, prev, k, m)
                    clause, is_reverse = res
                    return clause, k, m, None, is_reverse

        assert False
