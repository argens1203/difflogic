import logging
import logging

from pysat.card import CardEnc, EncType
from pysat.solvers import Solver


from lgn.util import feat_to_input

logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, encoding):
        self.encoding = encoding

        self.encoding = encoding
        self.votes_per_cls = self.encoding.get_votes_per_cls()
        self.clauses = dict()
        self.solvers = dict()

    def explain(self, feat):
        logger.debug("==== Explaining: %s ====", feat)

        inp = feat_to_input(feat)
        logger.info("Explaining Input: %s", inp)

        class_label = self.encoding.as_model()(feat.reshape(1, -1)).item()
        pred_class = class_label + 1

        logger.debug("Predicted Class - %s", pred_class)

        assert self.is_uniquely_satisfied_by(inp, pred_class)

        reduced = self.reduce_input(inp, pred_class)
        logger.info("One AXP: %s", reduced)

    def reduce_input(self, inp, predicted_cls):
        tmp_input = inp.copy()
        for feature in inp:
            logger.debug("Testing removal of %d", feature)
            tmp_input.remove(feature)
            if self.is_uniquely_satisfied_by(
                inp=tmp_input, predicted_cls=predicted_cls
            ):
                continue
            else:
                tmp_input.append(feature)
        return tmp_input

    def is_uniquely_satisfied_by(
        self,
        inp,
        predicted_cls,  # true_class as in not true class of data, but that predicted by model
    ):  # Return true means that only the true_class can satisfy all contraints given the input
        for cls in self.encoding.get_classes():
            if cls == predicted_cls:
                continue
            if (
                self.pairwise_comparisons(
                    true_class=predicted_cls, adj_class=cls, inp=inp
                )
                is True
            ):
                logger.debug("Satisfied by %d", cls)
                return False
        return True

    def get_clauses(self, true_class, adj_class):
        if (true_class, adj_class) in self.clauses:
            logger.debug(
                "Cached Clauses: %s", str(self.clauses[(true_class, adj_class)])
            )
            return self.clauses[(true_class, adj_class)]

        with self.encoding.use_context() as vpool:
            pos = self.encoding.get_output_ids(adj_class)
            neg = [-a for a in self.encoding.get_output_ids(true_class)]
            clauses = pos + neg  # Sum of X_i - Sum of X_pi_i > bounding number

            logger.debug("Lit: %s", str(clauses))

            bound = self.votes_per_cls + (1 if true_class < adj_class else 0)
            logger.debug("Bound: %d", bound)

            comp = CardEnc.atleast(
                lits=clauses,
                bound=bound,
                encoding=EncType.totalizer,
                vpool=vpool,
            )

            clauses = comp.clauses
            logger.debug("Recalculated Clauses: %s", str(comp.clauses))
            self.clauses[(true_class, adj_class)] = clauses
            return clauses

    def pairwise_comparisons(self, true_class, adj_class, inp=None):
        logger.debug(
            "==== Pairwise Comparisons (%d > %d) ====",
            true_class,
            adj_class,
        )

        with self.encoding.use_context() as vpool:
            clauses = self.get_clauses(true_class, adj_class)
            # Enumerate all clauses
            solver = self.get_solver(true_class, adj_class)

            solver.append_formula(clauses)
            logger.debug("Input: %s", inp)

            result = solver.solve(assumptions=inp)
            logger.debug("Satisfiable: %s", result)
            return result

    def get_solver(self, true_class, adj_class):
        if (true_class, adj_class) in self.solvers:
            logger.debug("Cached Solver")
            return self.solvers[(true_class, adj_class)]
        logger.info("Creating new solver (%d, %d)", true_class, adj_class)
        solver = Solver(bootstrap_with=self.encoding.cnf.clauses)
        solver.append_formula(self.get_clauses(true_class, adj_class))
        self.solvers[(true_class, adj_class)] = solver
        return solver

    def __del__(self):
        for _, solver in self.solvers.items():
            solver.delete()
        logger.info("Deleted all solvers")
