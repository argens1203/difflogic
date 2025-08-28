import logging
from tqdm import tqdm
from pysat.formula import Atom

from experiment.helpers.sat_context import SatContext
from lgn.dataset.auto_transformer import AutoTransformer

from .util import _get_layers, get_eq_constraints

logger = logging.getLogger(__name__)

from pysat.solvers import Solver as BaseSolver


class SatDeduplicator:
    def __init__(self, e_ctx):
        self.e_ctx = e_ctx
        self.context = SatContext()

    def _get_inputs(self, Dataset: AutoTransformer):
        with self.use_context() as vpool:
            input_handles = [Atom(i + 1) for i in range(Dataset.get_input_dim())]
            input_ids = [vpool.id(h) for h in input_handles]
        return input_handles, input_ids

    def _get_eq_constraints(self, input_ids):
        with self.use_context() as vpool:
            return get_eq_constraints(
                self.e_ctx.get_dataset(),
                input_ids,
                enc_type=self.e_ctx.get_enc_type(),
                vpool=vpool,
            )

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

    def deduplicate_c(self, gate):
        # print("gates", gates)
        if not self.solver.solve(assumptions=[-gate]):
            # print(solver.get_core())
            # print("is constant True", gate)
            self._add_clause([gate])
            return True
        if not self.solver.solve(assumptions=[gate]):
            # print(solver.get_core())
            # print("is constant False", gate)
            self._add_clause([-gate])
            return False
        return None

    def deduplicate_pair(self, i, j, gates):
        gate = gates[i][j]
        for k, layer in enumerate(gates):
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    return None, None, None
                is_reverse = self.dedup_pair_c(gate, prev)
                if is_reverse is not None:
                    return k, m, is_reverse

        assert False

    def _get_gates(self, input_ids, model):
        prev = input_ids
        gates = [input_ids]

        with self.use_context() as vpool:
            for layer in _get_layers(model):
                aux_vars = [vpool._next() for _ in range(layer.out_dim)]
                for f in layer.get_clauses(prev, aux_vars):
                    self._extend_clauses(f)
                gates.append(aux_vars)
                prev = aux_vars
        return gates

    def _add_clause(self, clause: list[int]):
        self._extend_clauses([clause])

    def _extend_clauses(self, clauses: list[list[int]]):
        self.clauses.extend(clauses)
        # print("self.clauses", len(self.clauses))
        # num_vars = max(abs(literal) for clause in self.clauses for literal in clause)
        # print("num_vars", num_vars)

        self.solver.append_formula(clauses)

    def _get_lookups(self, gates):
        const_lookup = dict()
        is_rev_lookup = dict()
        pair_lookup = dict()
        for i, layer_of_gates in enumerate(gates):
            for j, _ in tqdm(enumerate(layer_of_gates), total=len(layer_of_gates)):
                is_constant = self.deduplicate_c(gates[i][j])
                if is_constant is not None:
                    const_lookup[(i, j)] = is_constant
                    continue

                # is_reverse = self.deduplicate_input(
                #     gates[i][j],
                # )

                i_, j_, is_reverse = self.deduplicate_pair(i, j, gates)
                if is_reverse is not None:
                    is_rev_lookup[(i, j)] = is_reverse
                if i_ is not None and j_ is not None:
                    pair_lookup[(i, j)] = (i_, j_)
                else:
                    pair_lookup[(i, j)] = (i, j)
        return const_lookup, is_rev_lookup, pair_lookup

    def _initialize_solver(self, eq_constraints):
        solver = BaseSolver(name=self.e_ctx.get_solver_type())
        solver.append_formula(eq_constraints.clauses)  # OHE
        return solver

    def deduplicate(self, model, Dataset):
        input_handles, input_ids = self._get_inputs(Dataset)

        eq_constraints = self._get_eq_constraints(input_ids)
        self.solver = self._initialize_solver(eq_constraints)

        self.clauses = []
        gates = self._get_gates(input_ids, model)
        const_lookup, is_rev_lookup, pair_lookup = self._get_lookups(gates)

        return const_lookup, is_rev_lookup, pair_lookup

    def use_context(self):
        return self.context.use_vpool()
