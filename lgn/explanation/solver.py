from lgn.encoding import Encoding

import logging
from contextlib import contextmanager

from pysat.solvers import Solver as BaseSolver
from pysat.card import CardEnc, EncType
from pysat.formula import Formula
from experiment.helpers import Context

logger = logging.getLogger(__name__)


class Solver:
    def __init__(self, encoding: Encoding, ctx: Context):
        self.solver = BaseSolver(
            name=ctx.get_solver_type()
        )  # g42, cd19 > m22 # TODO: try other solvers
        self.encoding = encoding
        self._append_formula(encoding.get_cnf_clauses())
        # print("cnf", encoding.get_cnf_clauses())
        # NEW
        self._append_formula(encoding.get_eq_constraints_clauses())
        # print("eq_constraints", encoding.get_eq_constraints_clauses())

        self.vpool_context = encoding.s_ctx.get_vpool_context()
        # NEW
        # Formula.attach_vpool(self._copy_vpool(encoding), id(self))
        self.enc_type = ctx.get_enc_type()

    def set_cardinality(self, lits, bound):
        with self.use_context() as vpool:
            comp = CardEnc.atleast(
                lits=lits,
                bound=bound,
                encoding=self.enc_type,
                vpool=vpool,
            )
            clauses = comp.clauses
            # logger.debug("Clauses: %s", str(comp.clauses))
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

    @contextmanager
    def use_context(self):
        prev = Formula._context
        try:
            Formula.set_context(self.vpool_context)
            yield Formula.export_vpool(active=True)
        finally:
            Formula.set_context(prev)

    def __del__(self):
        self.delete()

    def delete(self):
        self.solver.delete()
        # Sovler rides on Encoding vpool, so we don't need to delete it
        # Formula.cleanup(id(self))

    def get_clause_count(self):
        return self.solver.nof_clauses()

    def get_var_count(self):
        return self.solver.nof_vars()
