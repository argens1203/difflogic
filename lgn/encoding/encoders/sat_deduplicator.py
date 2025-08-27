import logging
from pysat.formula import Formula, Atom, Or, Neg, CNF

from typing import Set

logger = logging.getLogger(__name__)


class DeduplicationMixin:
    def dedup_pair_c(self, gate, prev):
        with self.use_context() as vpool:
            auxvar_id = vpool._next()

            self.solver.append_formula([[auxvar_id, -gate, prev]])
            self.solver.append_formula([[auxvar_id, gate, -prev]])
            self.solver.append_formula([[-auxvar_id, -gate, -prev]])
            self.solver.append_formula([[-auxvar_id, gate, prev]])

            if not self.solver.solve(assumptions=[-auxvar_id]):
                self.solver.append_formula([[auxvar_id]])
                return True
            if not self.solver.solve(assumptions=[auxvar_id]):
                self.solver.append_formula([[-auxvar_id]])
                return False

    def deduplicate_c(self, i, j, gates):
        # print("gates", gates)
        gate = gates[i][j]
        if not self.solver.solve(assumptions=[-gate]):
            # print(solver.get_core())
            # print("is constant True", gate)
            self._add_clause([gate])
            return None, None, True, None
        if not self.solver.solve(assumptions=[gate]):
            # print(solver.get_core())
            # print("is constant False", gate)
            self._add_clause([-gate])
            return None, None, False, None

        for k, layer in enumerate(gates):
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    return None, None, None, None
                is_reverse = self.dedup_pair_c(gate, prev)
                if is_reverse is not None:
                    return k, m, None, is_reverse

        assert False
