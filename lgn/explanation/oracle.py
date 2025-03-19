import logging
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver

from lgn.encoding import Encoding

logger = logging.getLogger(__name__)


class SatOracle:
    def __init__(self, encoding: Encoding):
        self.encoding = encoding
        self.votes_per_cls = self.encoding.get_votes_per_cls()
        self.clauses = dict()

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
            with Solver(bootstrap_with=[]) as solver:
                solver.append_formula(clauses)
                logger.debug("Input: %s", inp)

                result = solver.solve(assumptions=inp)
                logger.debug("Satisfiable: %s", result)
                return result
