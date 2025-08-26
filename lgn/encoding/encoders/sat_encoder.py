from tqdm import tqdm
import logging
import torch

from difflogic import LogicLayer, GroupSum

from pysat.formula import Atom, Neg
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

    def get_layers(self, model) -> list[LogicLayer]:
        layers = []
        for layer in model:
            assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
            if isinstance(layer, GroupSum):
                continue
            layers.append(layer)
        return layers

    def stuff(self, input_handles, model, clauses):
        with self.use_context() as vpool:
            input_ids = [vpool.id(h) for h in input_handles]

        prev = input_ids
        gates = []
        clauses = []

        with self.use_context() as vpool:
            for layer in self.get_layers(model):
                aux_vars = [vpool._next() for _ in range(layer.out_dim)]
                for f in layer.get_clauses(prev, aux_vars):
                    clauses.extend(f)
                gates.append(aux_vars)
                prev = aux_vars

        # self.e_ctx.debug(lambda: print("clauses", clauses, "gates", gates))
        self.solver.append_formula(clauses)

        curr = input_handles
        lookup = dict()
        for i, layer in enumerate(self.get_layers(model)):
            curr = layer.get_formula(curr)
            special = {}
            for j, g in enumerate(curr):
                # self.e_ctx.debug(lambda: print("lookup", lookup))
                lookup[(i, j)] = g
                # self.e_ctx.debug(lambda: print("g", g))
                # self.e_ctx.debug(lambda: print("i", i, "j", j))
                clause, i_, j_, is_constant, is_reverse = self.deduplicate_c(
                    i, j, gates, self.solver, input_ids
                )
                # self.e_ctx.debug(lambda: print("clause", clause))
                # self.e_ctx.debug(lambda: print("i_", i_, "j_", j_))
                if clause is not None:
                    self.solver.append_formula([clause])
                    clauses.extend([clause])
                if is_constant is not None:
                    lookup[(i, j)] = Atom(is_constant)
                    curr[j] = lookup[(i, j)]
                    special[j] = lookup[(i, j)]
                    # self.e_ctx.debug(lambda: print("after", curr[j]))
                    # self.e_ctx.debug(lambda: input("Press Enter to continue..."))
                    if g != Atom(True) and g != Atom(False):
                        self.e_ctx.inc_deduplication()
                    continue
                if is_reverse is not None:
                    if is_reverse:
                        lookup[(i, j)] = Neg(lookup[(i_, j_)])
                    else:
                        lookup[(i, j)] = lookup[(i_, j_)]
                    self.e_ctx.inc_deduplication()
                curr[j] = lookup[(i, j)]
                # self.e_ctx.debug(lambda: print("after", curr[j]))
                # self.e_ctx.debug(lambda: input("Press Enter to continue..."))

        formula = curr
        output_ids = gates[-1]
        return formula, clauses, output_ids, special

    def get_encoding(self, model, Dataset: AutoTransformer):
        self.context = SatContext()

        clauses = []

        input_handles, input_ids = self.get_inputs(Dataset)

        self.solver, eq_constraints = self.initialize_solver(input_ids)

        formula, clauses, output_ids, special = self.stuff(
            input_handles, model, clauses
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
