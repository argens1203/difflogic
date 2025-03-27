import logging

from pysat.card import CardEnc, EncType
from pysat.examples.hitman import Hitman

from lgn.encoding import Encoding
from lgn.util import feat_to_input, input_to_feat

from .solver import Solver

logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, encoding: Encoding):
        self.encoding = encoding

        self.encoding = encoding
        self.votes_per_cls = self.encoding.get_votes_per_cls()
        self.solvers = dict()

    def explain(self, feat=None, inp=None):
        if inp is None:
            inp = feat_to_input(feat)
        if feat is None:
            feat = input_to_feat(inp)

        logger.info("\n")
        logger.info("Explaining Input: %s", inp)

        class_label = self.encoding.as_model()(feat.reshape(1, -1)).item()
        pred_class = class_label + 1

        logger.debug("Predicted Class - %s", pred_class)

        is_uniquely_satisfied, model = self.is_uniquely_satisfied_by(inp, pred_class)
        assert is_uniquely_satisfied, "Assertion Error: " + ",".join(map(str, inp))

        axp = self.get_one_axp(inp, pred_class)
        logger.info("One AXP: %s", axp)

    def get_one_axp(self, inp, predicted_cls):
        """
        Get one AXP for the input and predicted class

        :param inp: input features
        :param predicted_cls: predicted class

        :return: AXP
        """
        tmp_input = inp.copy()
        for feature in inp:
            logger.debug("Testing removal of input %d", feature)
            tmp_input.remove(feature)
            is_uniquely_satisfied, _ = self.is_uniquely_satisfied_by(
                inp=tmp_input, predicted_cls=predicted_cls
            )
            if is_uniquely_satisfied:
                continue
            else:
                tmp_input.append(feature)
        return tmp_input

    def is_uniquely_satisfied_by(
        self,
        inp,
        predicted_cls,  # true_class as in not true class of data, but that predicted by model
    ):  # Return true means that only the true_class can satisfy all contraints given the input
        """
        Check if the input is uniquely satisfied by the predicted class

        :param inp: input features
        :param predicted_cls: predicted class

        :return: True if the input is uniquely satisfied by the predicted class
        """
        for cls in self.encoding.get_classes():
            if cls == predicted_cls:
                continue
            is_satisfiable, model = self.is_adj_class_satisfiable(
                true_class=predicted_cls, adj_class=cls, inp=inp
            )
            if is_satisfiable:
                logger.debug("Satisfied by %d", cls)
                return False, model
        return True, None

    def is_adj_class_satisfiable(self, true_class, adj_class, inp=None):
        solver = self.get_solver(true_class, adj_class)
        is_satisfiable = solver.solve(assumptions=inp)

        logger.debug(
            "Adj class %d having more votes than %d is %spossible with input %s",
            adj_class,
            true_class,
            "" if is_satisfiable else "NOT ",
            str(inp),
        )
        return is_satisfiable, solver.get_model()

    def get_lits(self, true_class, adj_class):
        pos = self.encoding.get_output_ids(adj_class)
        neg = [-a for a in self.encoding.get_output_ids(true_class)]
        lits = pos + neg  # Sum of X_i - Sum of X_pi_i > bounding number

        logger.debug(
            "Lit(%d %s %d): %s",
            adj_class,
            ">" if true_class < adj_class else ">=",
            true_class,
            str(lits),
        )
        return lits

    def get_bound(self, true_class, adj_class):
        bound = self.votes_per_cls + (1 if true_class < adj_class else 0)

        logger.debug("Bound: %d", bound)
        return bound

    def get_solver(self, true_class, adj_class):
        if (true_class, adj_class) in self.solvers:
            Stat.inc_cache_hit(Cached.SOLVER)
            return self.solvers[(true_class, adj_class)]

        Stat.inc_cache_miss(Cached.SOLVER)

        lits = self.get_lits(true_class, adj_class)
        bound = self.get_bound(true_class, adj_class)

        solver = Solver(encoding=self.encoding)

        solver.set_cardinality(lits, bound)

        self.solvers[(true_class, adj_class)] = solver

        return solver

    def is_satisfiable(self, pred_class, inp):
        logger.debug("Checking satisfiability of %s", str(inp))
        is_uniquely_satsified, model = self.is_uniquely_satisfied_by(inp, pred_class)
        return not is_uniquely_satsified, model

    def mhs_mus_enumeration(self, inp=None, feat=None, xnum=1000, smallest=False):
        """
        Enumerate subset- and cardinality-minimal explanations.
        """
        logger.debug("Starting mhs_mus_enumeration")

        if inp is None:
            inp = feat_to_input(feat)

        class_label = self.encoding.as_model()(input_to_feat(inp).reshape(1, -1)).item()
        pred_class = class_label + 1
        # result
        expls = []

        # just in case, let's save dual (contrastive) explanations
        duals = []

        with Hitman(
            bootstrap_with=[inp], htype="sorted" if smallest else "lbx"
        ) as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(inp):
                is_satisfiable, model = self.is_satisfiable(
                    pred_class, inp=inp[:i] + inp[(i + 1) :]
                )
                if is_satisfiable:
                    hitman.hit([hypo])  # Add unit-size MCS
                    duals.append([hypo])  # Add unit-size MCS to duals

            # main loop
            itr = 0
            while True:
                hset = hitman.get()  # Get candidate MUS
                itr += 1

                logger.debug("itr: %s", itr)
                logger.debug("cand: %s", hset)

                if hset == None:
                    break

                is_satisfiable, model = self.is_satisfiable(pred_class, inp=hset)
                logger.debug("Model: %s", model)
                # test_sat, _ = self.is_satisfiable(inp=model)
                # assert test_sat, "Assertion Error: " + ",".join(map(str, model))
                if is_satisfiable:
                    logger.debug("IS satisfied %s", hset)
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(inp).difference(set(hset)))

                    # model = self.oracle.get_model()
                    for h in removed:
                        if model[abs(h) - 1] != h:
                            unsatisfied.append(h)
                        else:
                            hset.append(h)

                    logger.debug("Unsatisfied: %s", unsatisfied)
                    logger.debug("Hset: %s", hset)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        is_satisfiable, _ = self.is_satisfiable(
                            pred_class, inp=hset + [h]
                        )
                        if is_satisfiable:
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    logger.debug("To hit: %s", to_hit)

                    hitman.hit(to_hit)

                    duals.append([to_hit])
                else:
                    logger.debug("Is NOT satisfied %s", hset)
                    # print("expl:", hset)

                    expls.append(hset)

                    if len(expls) != xnum:
                        hitman.block(hset)
                    else:
                        break
        return expls, duals

    def __del__(self):
        for _, solver in self.solvers.items():
            solver.delete()
        logger.debug("Deleted all solvers")
        logger.debug("Cache Hit: %s", str(Stat.cache_hit))
        logger.debug("Cache Miss: %s", str(Stat.cache_miss))


class Cached:
    SOLVER = "solver"


class Stat:
    cache_hit = {Cached.SOLVER: 0}
    cache_miss = {Cached.SOLVER: 0}

    def inc_cache_hit(flag: str):
        Stat.cache_hit[flag] += 1

    def inc_cache_miss(flag: str):
        Stat.cache_miss[flag] += 1
