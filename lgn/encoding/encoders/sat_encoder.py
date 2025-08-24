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

    def _stuff(self, input_handles, model, clauses):
        all = OrderedSet()
        for i in input_handles:
            all.add(i)

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
                # print("aux_vars", aux_vars)
                # print("layer", layer)
                # layer.print()
                for f in layer.get_clauses(prev, aux_vars):
                    # print("clause", f)
                    clauses.extend(f)
                gates.append(aux_vars)
                # input("Press Enter to continue...")
                prev = aux_vars

        # print("clauses", clauses)
        # print("gates", gates)
        self.solver.append_formula(clauses)

        curr = input_handles
        lookup = dict()
        for i, layer in enumerate(self.get_layers(model)):
            curr = layer.get_formula(curr)
            for j, g in enumerate(curr):
                # print("lookup", lookup)
                lookup[(i, j)] = g
                # print("before", g)
                # print("i,", i, "j", j)
                clause, i_, j_, is_constant, is_reverse = self.deduplicate_c(
                    i, j, gates, self.solver, input_ids
                )
                # print("clause", clause)
                # print("i_", i_, "j_", j_)
                if clause is not None:
                    self.solver.append_formula([clause])
                    clauses.extend([clause])
                if is_constant is not None:
                    lookup[(i, j)] = Atom(is_constant)
                    curr[j] = lookup[(i, j)]
                    # print("after", curr[j])
                    # input("Press Enter to continue...")
                    continue
                if is_reverse is not None:
                    if is_reverse:
                        lookup[(i, j)] = Neg(lookup[(i_, j_)])
                    else:
                        lookup[(i, j)] = lookup[(i_, j_)]
                curr[j] = lookup[(i, j)]
                # print("after", curr[j])
                # input("Press Enter to continue...")

        formula = curr
        output_ids = gates[-1]
        special = {}
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
