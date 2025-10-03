from __future__ import annotations
import logging
from tqdm import tqdm
from pysat.formula import Atom

from experiment.helpers.sat_context import SatContext
from lgn.dataset.auto_transformer import AutoTransformer
from lgn.encoding.util import get_parts

from .util import _get_layers, get_eq_constraints

logger = logging.getLogger(__name__)

from pysat.solvers import Solver as BaseSolver

from typing import List, Optional


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

    def __deduplicate_pair_normal(self, i, j, vars, gates, deduplicate_ohe: bool):
        curr = vars[i][j]

        for k, layer in enumerate(vars):
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    return None, None, None
                if gates[i][j].ohe & gates[k][m].ohe == set():
                    continue
                is_reverse = self.dedup_pair_c(curr, prev)
                if is_reverse is not None:
                    if deduplicate_ohe and gates[i][j].ohe ^ gates[k][m].ohe != set():
                        if len(gates[i][j].ohe) < len(gates[k][m].ohe):
                            self.e_ctx.inc_ohe_deduplication(
                                gates[i][j].ohe, gates[k][m].ohe
                            )
                            gates[i][j].ohe = gates[k][m].ohe
                    return k, m, is_reverse

        assert False

    def __deduplicate_pair_reverse(self, i, j, vars, gates, deduplicate_ohe: bool):
        curr = vars[i][j]

        for k, layer in reversed(list(enumerate(vars))):
            if k > i:
                continue
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    break  # For reverse order, skip checking self and everything after in the same layer
                if gates[i][j].ohe & gates[k][m].ohe == set():
                    continue
                is_reverse = self.dedup_pair_c(curr, prev)
                if is_reverse is not None:
                    if deduplicate_ohe and gates[i][j].ohe ^ gates[k][m].ohe != set():
                        if len(gates[i][j].ohe) < len(gates[k][m].ohe):
                            self.e_ctx.inc_ohe_deduplication(
                                gates[i][j].ohe, gates[k][m].ohe
                            )
                            gates[i][j].ohe = gates[k][m].ohe
                    return k, m, is_reverse

        return None, None, None

    def __deduplicate_pair_one_layer(self, i, j, vars, gates, deduplicate_ohe: bool):
        curr = vars[i][j]
        for k, layer in reversed(list(enumerate(vars))):
            if k > i or (k < i - 1 and k != 0):
                continue
            # Only check current, previous and input layer
            for m, prev in enumerate(layer):
                if k == i and m == j:
                    break  # For reverse order, skip checking self and everything after in the same layer
                if gates[i][j].ohe & gates[k][m].ohe == set():
                    continue
                is_reverse = self.dedup_pair_c(curr, prev)
                if is_reverse is not None:
                    if deduplicate_ohe and gates[i][j].ohe ^ gates[k][m].ohe != set():
                        if len(gates[i][j].ohe) < len(gates[k][m].ohe):
                            self.e_ctx.inc_ohe_deduplication(
                                gates[i][j].ohe, gates[k][m].ohe
                            )
                            gates[i][j].ohe = gates[k][m].ohe
                    return k, m, is_reverse

        return None, None, None

    def deduplicate_pair(self, i, j, vars, gates, strategy: str, deduplicate_ohe: bool):
        if strategy == "full":
            return self.__deduplicate_pair_normal(
                i, j, vars=vars, gates=gates, deduplicate_ohe=deduplicate_ohe
            )
        elif strategy == "b_full":
            return self.__deduplicate_pair_reverse(
                i, j, vars=vars, gates=gates, deduplicate_ohe=deduplicate_ohe
            )
        elif strategy == "parent":
            return self.__deduplicate_pair_one_layer(
                i, j, vars=vars, gates=gates, deduplicate_ohe=deduplicate_ohe
            )

        assert False

    def __get_input_gates(self, input_ids: List[int]) -> List[Gate]:
        def __get_ohe_set(input_id: int, parts: List[List[int]]) -> set[int]:
            for part in parts:
                if input_id in part:
                    return set(part)
            assert False

        parts = get_parts(self.e_ctx.get_dataset(), input_ids)
        ret = []
        for i in range(len(input_ids)):
            ret.append(Gate(0, i, -1, __get_ohe_set(input_ids[i], parts), None))
        return ret

    def _get_gates(self, input_ids, model):
        gates: List[List[Gate]] = [self.__get_input_gates(input_ids)]

        for x, layer in enumerate(_get_layers(model)):
            curr_layer = []
            for idx, (op, a, b) in enumerate(layer.get_raw()):
                # logger.info(f"op: {op}, a: {a}, b: {b}")
                curr_layer.append(Gate(x + 1, idx, op, None, gates[x][a], gates[x][b]))
            gates.append(curr_layer)
        return gates

    def _get_vars(self, input_ids, model):
        prev = input_ids
        vars = [input_ids]

        with self.use_context() as vpool:
            for layer in _get_layers(model):
                aux_vars = [vpool._next() for _ in range(layer.out_dim)]
                for f in layer.get_clauses(prev, aux_vars):
                    self._extend_clauses(f)
                vars.append(aux_vars)
                prev = aux_vars
        return vars

    def _add_clause(self, clause: list[int]):
        self._extend_clauses([clause])

    def _extend_clauses(self, clauses: list[list[int]]):
        self.clauses.extend(clauses)
        # print("self.clauses", len(self.clauses))
        # num_vars = max(abs(literal) for clause in self.clauses for literal in clause)
        # print("num_vars", num_vars)

        self.solver.append_formula(clauses)

    def _get_lookups(self, vars, gates, strategy: str, deduplicate_ohe: bool):
        const_lookup = dict()
        is_rev_lookup = dict()
        pair_lookup = dict()

        for i, layer_of_gates in enumerate(vars):
            for j, _ in tqdm(enumerate(layer_of_gates), total=len(layer_of_gates)):
                is_constant = self.deduplicate_c(vars[i][j])
                if is_constant is not None:
                    const_lookup[(i, j)] = is_constant
                    continue
                i_, j_, is_reverse = self.deduplicate_pair(
                    i,
                    j,
                    vars=vars,
                    gates=gates,
                    strategy=strategy,
                    deduplicate_ohe=deduplicate_ohe,
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

    def deduplicate(self, model, Dataset, opt=dict()):
        strategy = opt.get("strategy", "full")
        deduplicate_ohe = opt.get("deduplicate_ohe", True)

        input_handles, input_ids = self._get_inputs(Dataset)

        eq_constraints = self._get_eq_constraints(input_ids)
        self.solver = self._initialize_solver(eq_constraints)

        self.clauses = []
        vars = self._get_vars(input_ids, model)
        gates = self._get_gates(input_ids, model)
        const_lookup, is_rev_lookup, pair_lookup = self._get_lookups(
            vars=vars, strategy=strategy, gates=gates, deduplicate_ohe=deduplicate_ohe
        )

        return const_lookup, is_rev_lookup, pair_lookup

    def use_context(self):
        return self.context.use_vpool()
