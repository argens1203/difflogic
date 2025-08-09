from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lgn.encoding import Encoding

import logging

from lgn.util import remove_none, Stat, Cached

from .solver import Solver

logger = logging.getLogger(__name__)


class MulticlassSolver:
    def __init__(self, encoding: Encoding):
        self.solvers = dict()
        self.encoding = encoding
        self.votes_per_cls = self.encoding.get_votes_per_cls()

    ## -- Public -- #
    def is_uniquely_satisfied_by(
        self,
        inp: list[int],  # input features
        predicted_cls,  # true_class as in not true class of data, but that predicted by model
    ):  # Return true means that only the true_class can satisfy all contraints given the input
        """
        Check if the input is uniquely satisfied by the predicted class

        :param inp: input features
        :param predicted_cls: predicted class

        :return: True if the input is uniquely satisfied by the predicted class
        """
        cores = []
        for cls in self.encoding.get_classes():
            if cls == predicted_cls:
                continue
            is_satisfiable, model, core = self.is_adj_class_satisfiable(
                true_class=predicted_cls, adj_class=cls, inp=inp
            )
            if is_satisfiable:
                logger.debug("Satisfied by %d", cls)
                return False, model, None
            else:
                cores.append(core)
        combined_core = set()
        for core in cores:
            combined_core = combined_core.union(set(core))
        return True, None, list(combined_core)

    def is_adj_class_satisfiable(self, true_class, adj_class, inp: list[int]):
        solver = self.get_solver(true_class, adj_class)
        is_satisfiable = solver.solve(assumptions=inp)

        logger.debug(
            "Adj class %d having more votes than %d is %spossible with input %s",
            adj_class,
            true_class,
            "" if is_satisfiable else "NOT ",
            str(inp),
        )
        return is_satisfiable, solver.get_model(), solver.get_core()

    def is_satisfiable(self, pred_class, inp: set[int]):
        is_satisfiable, _, __ = self.is_satisfiable_with_model_or_core(pred_class, inp)
        return is_satisfiable

    def is_satisfiable_with_model_or_core(self, pred_class, inp: list[int]):
        logger.debug("Checking satisfiability of %s", str(inp))
        is_uniquely_satsified, model, core = self.is_uniquely_satisfied_by(
            inp, pred_class
        )
        return not is_uniquely_satsified, model, core

    def solve(self, pred_class, inp: list[int]):
        # NEW
        self.assert_input_correctness(inp)
        # NEW
        solvable, model, core = self.is_satisfiable_with_model_or_core(pred_class, inp)
        return {
            "solvable": solvable,
            "model": model,
            "core": core,
        }

    def is_solvable(self, pred_class, inp: set[int]):
        # NEW
        self.assert_input_correctness(inp)
        # NEW
        return self.is_satisfiable(pred_class, inp)

    # # NEW
    def assert_input_correctness(self, inp):
        for part in self.encoding.get_parts():
            part_inp = list(filter(lambda x: x in part or -x in part, inp))
            # Skip / Allow if the entire part is not in the input
            if len(part_inp) == 0:
                continue

            filtered = list(filter(lambda x: x > 0, part_inp))
            assert len(filtered) == 1

    # # NEW

    ## -- Private -- #
    def get_solver(self, true_class, adj_class):
        if (true_class, adj_class) in self.solvers:
            Stat.inc_cache_hit(Cached.SOLVER)
            return self.solvers[(true_class, adj_class)]

        Stat.inc_cache_miss(Cached.SOLVER)

        solver = Solver(encoding=self.encoding)

        lits, bound = self.get_lits_and_bound(true_class, adj_class)
        solver.set_cardinality(lits, bound)

        self.solvers[(true_class, adj_class)] = solver

        return solver

    def get_lits_and_bound(self, true_class, adj_class):
        pos, pos_none_idxs = remove_none(self.encoding.get_output_ids(adj_class))
        neg, neg_none_idxs = remove_none(self.encoding.get_output_ids(true_class))
        neg = [-a for a in neg]
        lits = pos + neg  # Sum of X_i - Sum of X_pi_i > bounding number

        logger.debug(
            "Lit(%d %s %d): %s",
            adj_class,
            ">" if true_class < adj_class else ">=",
            true_class,
            str(lits),
        )

        bound = self.votes_per_cls + (1 if true_class < adj_class else 0)

        for idx in pos_none_idxs:
            if self.encoding.get_truth_value(idx) is True:
                bound -= 1  # One vote less for each defenite True in the output of the adj class
        for idx in neg_none_idxs:
            if self.encoding.get_truth_value(idx) is False:
                bound -= 1  # One vote more is needed for each defenite True in the output of the true class

        logger.debug("Bound: %d", bound)
        return lits, bound

    def __del__(self):
        for _, solver in self.solvers.items():
            solver.delete()
        logger.debug("Deleted all solvers")
