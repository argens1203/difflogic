import logging
from contextlib import contextmanager

from pysat.solvers import Solver as BaseSolver
from pysat.card import CardEnc, EncType
from pysat.formula import Formula, IDPool

from lgn.encoding import Encoding

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self, encoding: Encoding):
        self.solver = BaseSolver()
        self.encoding = encoding
        self._append_formula(encoding.cnf.clauses)
        Formula.attach_vpool(id(self), self._copy_vpool(encoding))

    def set_cardinality(self, lits, bound):
        with self.use_context() as vpool:
            comp = CardEnc.atleast(
                lits=lits,
                bound=bound,
                encoding=EncType.totalizer,
                vpool=vpool,
            )
            clauses = comp.clauses
            logger.debug("Clauses: %s", str(comp.clauses))
            self._append_formula(clauses)
        return self

    def solve(self, assumptions=[]):
        return self.solver.solve(assumptions=assumptions)

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
