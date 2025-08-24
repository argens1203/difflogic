from tqdm import tqdm
import logging
import torch

from difflogic import LogicLayer, GroupSum

from pysat.formula import Atom
from pysat.card import EncType
from pysat.solvers import Solver as BaseSolver

from experiment.helpers.ordered_set import OrderedSet
from experiment.helpers import SatContext

from lgn.dataset import AutoTransformer

from .sat_deduplicator import DeduplicationMixin
from .encoder import Encoder
from ..encoding import Encoding


logger = logging.getLogger(__name__)


class SatEncoder(Encoder, DeduplicationMixin):
    def get_inputs(self, Dataset: AutoTransformer):
        with self.use_context() as vpool:
            input_handles = [Atom(i + 1) for i in range(Dataset.get_input_dim())]
            input_ids = [vpool.id(h) for h in input_handles]
        return input_handles, input_ids

    def initialize_solver(self, input_ids):
        with self.use_context() as vpool:
            eq_constraints = self.initialize_ohe(
                self.e_ctx.get_dataset(), input_ids, enc_type=self.e_ctx.get_enc_type()
            )
        solver = BaseSolver(name=self.e_ctx.get_solver_type())
        solver.append_formula(eq_constraints.clauses)  # OHE
        return solver, eq_constraints

    def _get_formula(self, model, input_handles):
        x = input_handles
        with self.use_context():
            for layer in model:
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):  # TODO: make get_formula for GroupSum
                    continue
                x = layer.get_formula(x)
        return x

    def stuff(self, input_handles, model, all, clauses):
        x = input_handles
        with self.use_context():
            for layer in model:
                this_layer = []
                assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
                if isinstance(layer, GroupSum):
                    continue
                for f in tqdm(layer.get_formula(x)):
                    f = self.deduplicate(f, all)
                    f.clausify()
                    this_layer.append(f)
                    if f != Atom(True) and f != Atom(False):
                        all.add(f)
                    clauses.extend(f.clauses)
                x = this_layer

            input_ids, cnf, output_ids, special = self.populate_clauses(
                input_handles=input_handles, formula=x
            )
        return x, cnf.clauses, output_ids, special

    def get_encoding(self, model, Dataset: AutoTransformer):
        self.context = SatContext()

        clauses = []

        input_handles, input_ids = self.get_inputs(Dataset)
        all = OrderedSet()
        for i in input_handles:
            all.add(i)

        self.solver, eq_constraints = self.initialize_solver(input_ids)

        formula, clauses, output_ids, special = self.stuff(
            input_handles, model, all, clauses
        )

        return Encoding(
            clauses=clauses,
            eq_constraints=eq_constraints,
            input_ids=input_ids,
            output_ids=output_ids,
            formula=formula,
            special=special,
            s_ctx=self.context,
            e_ctx=self.e_ctx,
        )
