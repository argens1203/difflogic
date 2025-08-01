from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lgn.encoding import Encoding

import logging
from contextlib import contextmanager

from pysat.solvers import Solver as BaseSolver
from pysat.card import CardEnc, EncType
from pysat.formula import Formula, IDPool

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self, encoding: Encoding):
        self.solver = BaseSolver()
        self.encoding = encoding
        self._append_formula(encoding.cnf.clauses)
        # NEW
        self._append_formula(encoding.eq_constraints.clauses)
        # NEW
        Formula.attach_vpool(self._copy_vpool(encoding), id(self))

        self.enc_type = encoding.get_enc_type()

    def set_cardinality(self, lits, bound):
        with self.use_context() as vpool:
            comp = CardEnc.atleast(
                lits=lits,
                bound=bound,
                encoding=self.enc_type,
                vpool=vpool,
            )
            clauses = comp.clauses
            logger.debug("Clauses: %s", str(comp.clauses))
            self._append_formula(clauses)
        return self

    def solve(self, assumptions: list[int] = []):
        return self.solver.solve(assumptions=assumptions)

    def get_model(self):
        model = self.solver.get_model()
        # NEW
        self.assert_model_correctness(model)
        # NEW
        return model

    # NEW
    def assert_model_correctness(self, model):
        def ensure_one_positive(part):
            assert len(list(filter(lambda x: x > 0, part))) == 1

        if model is None:
            return

        itr = 0
        for step in self.encoding.get_attribute_ranges():
            ensure_one_positive(model[itr : itr + step])
            itr += step

    # NEW

    def get_core(self):
        core = self.solver.get_core()

        return core

    def _append_formula(self, clauses=[]):
        self.solver.append_formula(clauses)
        return self

    def _copy_vpool(self, encoding: Encoding):
        with encoding.use_context() as vpool:
            id_pool = IDPool()
            id_pool.top = vpool.top
            id_pool.obj2id = vpool.obj2id.copy()
            id_pool.id2obj = vpool.id2obj.copy()
            id_pool._occupied = vpool._occupied.copy()
            return id_pool

    @contextmanager
    def use_context(self):
        hashable = id(self)
        prev = Formula._context
        try:
            Formula.set_context(hashable)
            yield Formula.export_vpool(active=True)
        finally:
            Formula.set_context(prev)

    def __del__(self):
        self.delete()

    def delete(self):
        self.solver.delete()
        Formula.cleanup(id(self))
