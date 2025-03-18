import logging
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver

from lgn.util import feat_to_input
from lgn.encoding import Encoding

logger = logging.getLogger(__name__)


class Explainer:
    def calc_bound(self, true_class, adj_class):
        # TODO: CHECK
        # If true_class < adj_class, then adj class need one more vote to be selected
        # Else the number of vote needed is output neuron / number of classes
        if true_class < adj_class:
            return len(self.output_ids) // self.class_dim + 1
        return len(self.output_ids) // self.class_dim

    def is_uniquely_satisfied_by(
        self,
        inp,
        predicted_cls,  # true_class as in not true class of data, but that predicted by model
    ):  # Return true means that only the true_class can satisfy all contraints given the input
        for cls in range(1, self.class_dim + 1):  # TODO: use more consistent ways
            if cls == predicted_cls:
                continue
            if (
                Explainer.pairwise_comparisons(
                    self, true_class=predicted_cls, adj_class=cls, inp=inp
                )
                is True
            ):
                logger.debug("Satisfied by %d", cls)
                return False
        return True

    def pairwise_comparisons(self: Encoding, true_class, adj_class, inp=None):
        logger.debug(
            "==== Pairwise Comparisons (%d > %d) ====",
            true_class,
            adj_class,
        )

        with self.use_context() as vpool:
            pos = self.get_output_ids(adj_class)
            neg = [-a for a in self.get_output_ids(true_class)]
            clauses = pos + neg  # Sum of X_i - Sum of X_pi_i > bounding number
            logger.debug("Lit: %s", str(clauses))

            bound = Explainer.calc_bound(self, true_class, adj_class)
            logger.debug("Bound: %d", bound)
            comp = CardEnc.atleast(
                lits=clauses,
                bound=bound,
                encoding=EncType.totalizer,
                vpool=vpool,
            )

            clauses = comp.clauses
            logger.debug("Card Encoding Clauses: %s", str(comp.clauses))

            # Enumerate all clauses
            with Solver(bootstrap_with=clauses) as solver:
                # TODO: CHECK
                # Check if it is satisfiable under cardinatlity constraint
                solver.append_formula(comp)
                logger.debug("Input: %s", inp)
                result = solver.solve(assumptions=inp)
                logger.debug("Satisfiable: %s", result)
                return result

        assert False, "Pairwise comparison error"

    def explain(self: Encoding, feat):
        logger.debug("==== Explaining: %s ====", feat)

        inp = feat_to_input(feat)
        logger.debug("inp: %s", inp)
        logger.info("Explaining: %s", inp)

        votes = self.predict_votes(feat.reshape(1, -1))
        logger.debug("Votes: %s", votes)

        true_class = votes.argmax().int() + 1
        logger.info("Predicted Class - %s", true_class)

        assert Explainer.is_uniquely_satisfied_by(self, inp, true_class)

        reduced = Explainer.reduce_input(self, inp, true_class)
        logger.info("Final reduced: %s", reduced)

    def reduce_input(self, inp, predicted_cls):
        tmp_input = inp.copy()
        for feature in inp:
            logger.info("Testing removal of %d", feature)
            tmp_input.remove(feature)
            if Explainer.is_uniquely_satisfied_by(
                self, inp=tmp_input, predicted_cls=predicted_cls
            ):
                continue
            else:
                tmp_input.append(feature)
        return tmp_input
