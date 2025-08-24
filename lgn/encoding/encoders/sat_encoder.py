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

    def get_encoding(self, model, Dataset: AutoTransformer):
        solver_type = self.e_ctx.get_solver_type()
        self.context = SatContext()

        clauses = []

        input_handles, input_ids = self.get_inputs(Dataset)
        all = OrderedSet()
        for i in input_handles:
            all.add(i)

        x = input_handles
        with self.use_context():
            eq_constraints = self.initialize_ohe(
                Dataset, input_ids, enc_type=self.e_ctx.get_enc_type()
            )
            self.solver = BaseSolver(name=solver_type)
            self.solver.append_formula(eq_constraints.clauses)  # OHE

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
            return Encoding(
                clauses=cnf.clauses,
                eq_constraints=eq_constraints,
                input_ids=input_ids,
                output_ids=output_ids,
                formula=x,
                special=special,
                s_ctx=self.context,
                e_ctx=self.e_ctx,
            )
