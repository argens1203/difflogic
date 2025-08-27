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
    def _get_inputs(self, Dataset: AutoTransformer):
        with self.use_context() as vpool:
            input_handles = [Atom(i + 1) for i in range(Dataset.get_input_dim())]
            input_ids = [vpool.id(h) for h in input_handles]
        return input_handles, input_ids

    def _initialize_solver(self, input_ids):
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

    def _get_layers(self, model) -> list[LogicLayer]:
        layers = []
        for layer in model:
            assert isinstance(layer, LogicLayer) or isinstance(layer, GroupSum)
            if isinstance(layer, GroupSum):
                continue
            layers.append(layer)
        return layers

    def _add_clause(self, clause: list[int]):
        self._extend_clauses([clause])

    def _extend_clauses(self, clauses: list[list[int]]):
        self.clauses.extend(clauses)
        print("self.clauses", len(self.clauses))
        num_vars = max(abs(literal) for clause in self.clauses for literal in clause)
        print("num_vars", num_vars)

        self.solver.append_formula(clauses)

    def _stuff(self, input_handles, model):
        self.clauses = []
        with self.use_context() as vpool:
            input_ids = [vpool.id(h) for h in input_handles]

        prev = input_ids
        gates = []

        with self.use_context() as vpool:
            for layer in self._get_layers(model):
                aux_vars = [vpool._next() for _ in range(layer.out_dim)]
                for f in layer.get_clauses(prev, aux_vars):
                    self._extend_clauses(f)
                gates.append(aux_vars)
                prev = aux_vars

        const_lookup = dict()
        is_rev_lookup = dict()
        pair_lookup = dict()
        for i, layer_of_gates in enumerate(gates):
            for j, _ in enumerate(layer_of_gates):
                i_, j_, is_constant, is_reverse = self.deduplicate_c(i, j, gates)
                if is_constant is not None:
                    const_lookup[(i, j)] = is_constant
                if is_reverse is not None:
                    is_rev_lookup[(i, j)] = is_reverse
                if i_ is not None and j_ is not None:
                    pair_lookup[(i, j)] = (i_, j_)
                else:
                    pair_lookup[(i, j)] = (i, j)

        curr = input_handles
        lookup = dict()
        for i, layer in enumerate(self._get_layers(model)):
            curr = layer.get_formula(curr)
            special = {}
            for j, g in enumerate(curr):
                if (i, j) in const_lookup:
                    lookup[(i, j)] = Atom(const_lookup[(i, j)])
                    curr[j] = lookup[(i, j)]
                    special[j] = lookup[(i, j)]
                    if g != Atom(True) and g != Atom(False):
                        self.e_ctx.inc_deduplication()
                elif (i, j) in is_rev_lookup:
                    if is_rev_lookup[(i, j)]:
                        lookup[(i, j)] = Neg(lookup[pair_lookup[(i, j)]])
                        curr[j] = lookup[(i, j)]
                    else:
                        lookup[(i, j)] = lookup[pair_lookup[(i, j)]]
                        curr[j] = lookup[(i, j)]
                    self.e_ctx.inc_deduplication()
                else:
                    lookup[(i, j)] = g

        formula = curr
        output_ids = gates[-1]
        return formula, output_ids, special

    def get_encoding(self, model, Dataset: AutoTransformer):
        self.context = SatContext()

        input_handles, input_ids = self._get_inputs(Dataset)

        self.solver, eq_constraints = self._initialize_solver(input_ids)

        formula, output_ids, special = self._stuff(input_handles, model)
        with self.use_context() as vpool:
            print("next vpool var", vpool._next())
            input("Press enter to continue...")

        return Encoding(
            clauses=self.clauses,
            eq_constraints=eq_constraints,
            input_ids=input_ids,
            output_ids=output_ids,
            formula=formula,
            special=special,
            s_ctx=self.context,
            e_ctx=self.e_ctx,
        )
