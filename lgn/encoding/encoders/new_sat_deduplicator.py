from __future__ import annotations
import logging
from tqdm import tqdm
from pysat.formula import Atom

from difflogic.functional import idx_to_clauses
from experiment.helpers.sat_context import SatContext
from lgn.dataset.auto_transformer import AutoTransformer
from lgn.encoding.util import get_parts
from .util import _get_layers, get_eq_constraints

logger = logging.getLogger(__name__)

from typing import Dict, List, Optional, Tuple
from pysat.solvers import Solver as BaseSolver


class Gate:

    x: int
    y: int
    op: int
    ohe: set[int]
    left: Optional[Gate]
    right: Optional[Gate]

    def __init__(
        self,
        x: int,
        y: int,
        op: int = -1,
        ohe: Optional[set[int]] = None,
        left: Optional[Gate] = None,
        right: Optional[Gate] = None,
    ):
        self.x = x
        self.y = y
        self.op = op
        self.left = left
        self.right = right

        # Op defintions
        __atom = [0, 15]
        __left = [3, 12]
        __right = [5, 10]
        __both = [1, 2, 4, 6, 7, 8, 9, 11, 13, 14]

        if ohe is None:
            assert left is not None and right is not None
            if op in __atom:
                self.ohe = set()
            elif op in __left:
                self.ohe = left.ohe
            elif op in __right:
                self.ohe = right.ohe
            elif op in __both:
                self.ohe = left.ohe | right.ohe
            else:
                assert False, f"Unknown op: {op}"
        else:
            self.ohe = ohe

    def __str__(self):
        if self.op == -1:
            return f"Gate({(self.x, self.y)}, INPUT, ohe={self.ohe})"
        return f"Gate({(self.x, self.y)}, op={self.op}, ohe={self.ohe})"


class NewSatDeduplicator:
    def __init__(self, e_ctx):
        self.e_ctx = e_ctx
        self.context = SatContext()

    def _get_inputs(self, Dataset: AutoTransformer):
        with self.use_context() as vpool:
            input_handles = [Atom(i + 1) for i in range(Dataset.get_input_dim())]
            input_ids = [vpool.id(h) for h in input_handles]
        return input_handles, input_ids

    def _get_eq_constraints(self, input_ids):
        parts = get_parts(self.e_ctx.get_dataset(), input_ids)
        # print(parts)
        with self.use_context() as vpool:
            return get_eq_constraints(
                self.e_ctx.get_dataset(),
                input_ids,
                enc_type=self.e_ctx.get_enc_type(),
                vpool=vpool,
            )

    def dedup_pair_c(self, gate, prev, input_ids):
        vars, clauses, max_id = self.__to_var([gate, prev], input_ids)
        assert len(vars) == 2
        gate, prev = vars
        eq_constraints = self._get_eq_constraints(input_ids)
        solver = self._initialize_solver(eq_constraints)
        solver.append_formula(clauses)

        auxvar_id = max_id + 1

        solver.append_formula([[auxvar_id, -gate, prev]])
        solver.append_formula([[auxvar_id, gate, -prev]])
        solver.append_formula([[-auxvar_id, -gate, -prev]])
        solver.append_formula([[-auxvar_id, gate, prev]])

        if not solver.solve(assumptions=[-auxvar_id]):
            solver.append_formula([[auxvar_id]])
            return True
        if not solver.solve(assumptions=[auxvar_id]):
            solver.append_formula([[-auxvar_id]])
            return False

    def __to_var(
        self, gates: List[Gate], input_ids: List[int]
    ) -> Tuple[List[int], List[List[int]], int]:
        # Converts a gate to clauses and number for SAT solver
        max_id = max(input_ids)
        var_list = input_ids[:]
        seen: Dict[Tuple[int, int], int] = dict()  # (x, y) -> var_id
        for i in range(len(var_list)):
            seen[(0, i)] = var_list[i]

        clauses = []

        def __get_var(g: Gate) -> int:
            nonlocal max_id
            if (g.x, g.y) in seen:
                return seen[(g.x, g.y)]
            assert g.left is not None and g.right is not None
            var_list.append(max_id + 1)
            aux_var = var_list[-1]
            max_id += 1
            cl = idx_to_clauses(__get_var(g.left), __get_var(g.right), g.op, aux_var)
            # input("Press Enter to continue...")
            clauses.extend(cl)
            seen[(g.x, g.y)] = aux_var
            return aux_var

        return [__get_var(gate) for gate in gates], clauses, max_id

    def deduplicate_c(self, gate, input_ids: List[int]):
        # print("gates", gates)
        gates, clauses, _ = self.__to_var([gate], input_ids)
        assert len(gates) == 1
        gate = gates[0]
        eq_constraints = self._get_eq_constraints(input_ids)
        solver = self._initialize_solver(eq_constraints)
        # print(solver.)
        # input("Press Enter to continue...")
        solver.append_formula(clauses)

        if not solver.solve(assumptions=[-gate]):
            # print(solver.get_core())
            # print("is constant True", gate)
            # self._add_clause([gate])
            return True
        if not solver.solve(assumptions=[gate]):
            # print(solver.get_core())
            # print("is constant False", gate)
            # self._add_clause([-gate])
            return False
        return None

    def __deduplicate_pair_normal(self, i, j, gates, input_ids):  # Updated signature
        gate = gates[i][j]

        for k, layer in enumerate(gates):
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    return None, None, None
                if gate.ohe & prev.ohe == set():
                    continue
                is_reverse = self.dedup_pair_c(gate, prev, input_ids)
                if is_reverse is not None:
                    return k, m, is_reverse

        assert False

    def __deduplicate_pair_reverse(self, i, j, gates):
        gate = gates[i][j]

        for k, layer in reversed(list(enumerate(gates))):
            if k > i:
                continue
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    break  # For reverse order, skip checking self and everything after in the same layer
                is_reverse = self.dedup_pair_c(gate, prev)
                if is_reverse is not None:
                    return k, m, is_reverse

        return None, None, None

    def __deduplicate_pair_one_layer(self, i, j, gates):
        gate = gates[i][j]
        for k, layer in reversed(list(enumerate(gates))):
            if k > i or (k < i - 1 and k != 0):
                continue
            # Only check current, previous and input layer
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    break  # For reverse order, skip checking self and everything after in the same layer
                is_reverse = self.dedup_pair_c(gate, prev)
                if is_reverse is not None:
                    return k, m, is_reverse

        return None, None, None

    def deduplicate_pair(self, i, j, gates, strategy: str, input_ids: List[int]):
        if strategy == "ohe":
            return self.__deduplicate_pair_normal(i, j, gates, input_ids)
        elif strategy == "b_full":
            return self.__deduplicate_pair_reverse(i, j, gates)
        elif strategy == "parent":
            return self.__deduplicate_pair_one_layer(i, j, gates)

        assert False

    def __get_ohe_set(self, input_id: int, parts: List[List[int]]) -> set[int]:
        for part in parts:
            if input_id in part:
                return set(part)
        assert False

    def __get_input_gates(self, input_ids: List[int]) -> List[Gate]:
        parts = get_parts(self.e_ctx.get_dataset(), input_ids)
        ret = []
        for i in range(len(input_ids)):
            ret.append(Gate(0, i, -1, self.__get_ohe_set(input_ids[i], parts), None))
        return ret

    def _get_gates(self, input_ids, model):
        gates: List[List[Gate]] = [self.__get_input_gates(input_ids)]

        with self.use_context() as vpool:
            for x, layer in enumerate(_get_layers(model)):
                curr_layer = []
                for idx, (op, a, b) in enumerate(layer.get_raw()):
                    # logger.info(f"op: {op}, a: {a}, b: {b}")
                    curr_layer.append(
                        Gate(x + 1, idx, op, None, gates[x][a], gates[x][b])
                    )
                gates.append(curr_layer)
        return gates

    def _add_clause(self, clause: list[int]):
        self._extend_clauses([clause])

    def _extend_clauses(self, clauses: list[list[int]]):
        self.clauses.extend(clauses)
        # print("self.clauses", len(self.clauses))
        # num_vars = max(abs(literal) for clause in self.clauses for literal in clause)
        # print("num_vars", num_vars)

        self.solver.append_formula(clauses)

    def _get_lookups(self, gates, strategy: str, input_ids: list[int]):
        const_lookup = dict()
        is_rev_lookup = dict()
        pair_lookup = dict()

        for i, layer_of_gates in enumerate(gates):
            for j, _ in tqdm(enumerate(layer_of_gates), total=len(layer_of_gates)):
                is_constant = self.deduplicate_c(gates[i][j], input_ids=input_ids)
                if is_constant is not None:
                    const_lookup[(i, j)] = is_constant
                    continue

                i_, j_, is_reverse = self.deduplicate_pair(
                    i, j, gates, strategy=strategy, input_ids=input_ids
                )
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

    def deduplicate(self, model, Dataset, strategy: str = "full"):
        input_handles, input_ids = self._get_inputs(Dataset)

        eq_constraints = self._get_eq_constraints(input_ids)
        # self.solver = self._initialize_solver(eq_constraints)

        # self.clauses = []
        gates = self._get_gates(input_ids, model)
        const_lookup, is_rev_lookup, pair_lookup = self._get_lookups(
            gates, strategy=strategy, input_ids=input_ids
        )

        return const_lookup, is_rev_lookup, pair_lookup

    def use_context(self):
        return self.context.use_vpool()
