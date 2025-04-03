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

        is_uniquely_satisfied, _, __ = self.is_uniquely_satisfied_by(inp, pred_class)
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
            is_uniquely_satisfied, _, __ = self.is_uniquely_satisfied_by(
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
        # print(cores)
        # input()
        combined_core = set()
        for core in cores:
            combined_core = combined_core.union(set(core))
        # print(combined_core)
        # input()
        return True, None, list(combined_core)

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
        return is_satisfiable, solver.get_model(), solver.get_core()

    def remove_none(self, lst):
        ret = []
        indices = []
        for idx, l in enumerate(lst):
            if l is not None:
                ret.append(l)
            else:
                indices.append(idx)
        return ret, indices

    def get_lits_and_bound(self, true_class, adj_class):
        pos, pos_none_idxs = self.remove_none(self.encoding.get_output_ids(adj_class))
        neg, neg_none_idxs = self.remove_none(self.encoding.get_output_ids(true_class))
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

    def get_solver(self, true_class, adj_class):
        if (true_class, adj_class) in self.solvers:
            Stat.inc_cache_hit(Cached.SOLVER)
            return self.solvers[(true_class, adj_class)]

        Stat.inc_cache_miss(Cached.SOLVER)

        lits, bound = self.get_lits_and_bound(true_class, adj_class)

        solver = Solver(encoding=self.encoding)

        solver.set_cardinality(lits, bound)

        self.solvers[(true_class, adj_class)] = solver

        return solver

    def is_satisfiable(self, pred_class, inp):
        is_satisfiable, _, __ = self.is_satisfiable_with_model_or_core(pred_class, inp)
        return is_satisfiable

    def is_satisfiable_with_model_or_core(self, pred_class, inp):
        logger.debug("Checking satisfiability of %s", str(inp))
        is_uniquely_satsified, model, core = self.is_uniquely_satisfied_by(
            inp, pred_class
        )
        return not is_uniquely_satsified, model, core

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
            logger.info("Starting mhs_mus_enumeration")
            logger.info("Input: %s", inp)
            itr = 0
            # computing unit-size MCSes
            for i, hypo in enumerate(inp):
                if self.is_satisfiable(pred_class, inp=inp[:i] + inp[(i + 1) :]):
                    itr += 1
                    hitman.hit([hypo])  # Add unit-size MCS
                    duals.append([hypo])  # Add unit-size MCS to duals

            # main loop
            while True:
                hset = hitman.get()  # Get candidate MUS
                itr += 1

                # logger.info("itr: %s", itr)
                logger.info("itr %s) cand: %s", itr, hset)

                if hset == None:  # Terminates when there is no more candidate MUS
                    break

                is_satisfiable, model, _ = self.is_satisfiable_with_model_or_core(
                    pred_class, inp=hset
                )
                logger.debug("Model: %s", model)
                # test_sat, _ = self.is_satisfiable(inp=model)
                # assert test_sat, "Assertion Error: " + ",".join(map(str, model))
                if is_satisfiable:
                    logger.debug("IS satisfied %s", hset)
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(
                        set(inp).difference(set(hset))
                    )  # CXP lies within removed features

                    # model = self.oracle.get_model()
                    for h in removed:
                        if (
                            model[abs(h) - 1] != h
                        ):  # If a feature(hypothesis) of the input is different from that of the "solution"(model)
                            unsatisfied.append(h)  # Add it to unsatisfied
                        else:
                            hset.append(h)  # Else append it to hset
                            # How can we be sure adding h to hset keeps hset satisfiable?

                    logger.debug("Unsatisfied: %s", unsatisfied)
                    logger.debug("Hset: %s", hset)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        if self.is_satisfiable(
                            pred_class, inp=hset + [h]
                        ):  # Keep adding while satisfiable
                            hset.append(h)
                        else:
                            to_hit.append(h)
                            # Partial MCS found in a reversed manner

                    logger.info("To hit: %s", to_hit)

                    hitman.hit(to_hit)  # the entirity of to_hit is a MCS

                    duals.append(to_hit)
                else:
                    logger.debug("Is NOT satisfied %s", hset)
                    # print("expl:", hset)

                    expls.append(hset)  # Minimum Unsatisfiable Subset found - AXP found

                    if len(expls) != xnum:
                        hitman.block(hset)
                    else:
                        break
        assert itr == (len(expls) + len(duals) + 1), "Assertion Error: " + ",".join(
            map(str, [itr, len(expls), len(duals)])
        )
        return expls, duals

    def __del__(self):
        for _, solver in self.solvers.items():
            solver.delete()
        logger.debug("Deleted all solvers")
        logger.debug("Cache Hit: %s", str(Stat.cache_hit))
        logger.debug("Cache Miss: %s", str(Stat.cache_miss))

    def mhs_mcs_enumeration(
        self,
        xnum=1000,
        smallest=False,
        reduce_="none",
        unit_mcs=False,
        inp=None,
        feat=None,
    ):
        """
        Enumerate subset- and cardinality-minimal contrastive explanations.
        """
        expls = []  # result
        duals = []  # just in case, let's save dual (abductive) explanations
        possibility = 0
        if inp is None:
            inp = feat_to_input(feat)

        class_label = self.encoding.as_model()(input_to_feat(inp).reshape(1, -1)).item()
        pred_class = class_label + 1

        with Hitman(
            bootstrap_with=[inp], htype="sorted" if smallest else "lbx"
        ) as hitman:
            itr = 0
            logger.info("Starting mhs_mcs_enumeration")
            # computing unit-size MUSes
            for i, hypo in enumerate(inp):
                if not self.is_satisfiable(pred_class=pred_class, inp=[hypo]):
                    itr += 1
                    hitman.hit([hypo])
                    duals.append([hypo])
                elif unit_mcs and self.is_satisfiable(
                    pred_class=pred_class, inp=inp[:i] + inp[(i + 1) :]
                ):
                    itr += 1
                    # this is a unit-size MCS => block immediately
                    hitman.block([hypo])
                    expls.append([hypo])

            # main loop
            while True:
                hset = hitman.get()
                itr += 1

                # logger.info("itr: %s", itr)
                # logger.debug("itr: %s", itr)
                logger.info("itr %s) cand: %s", itr, hset)

                if hset == None:
                    break

                is_satisfiable, model, core = self.is_satisfiable_with_model_or_core(
                    pred_class=pred_class,
                    inp=sorted(set(inp).difference(set(hset))),
                )
                if not is_satisfiable:
                    to_hit = core  # Core is a weak (non-minimal) AXP?

                    if len(to_hit) > 1:
                        possibility += 1

                    if len(to_hit) > 1:
                        to_hit = self.get_one_axp(inp=to_hit, predicted_cls=pred_class)
                        # to_hit = self.extract_mus(reduce_=reduce_, start_from=to_hit)

                    logger.info("to_hit: %s", to_hit)

                    duals.append(to_hit)
                    hitman.hit(to_hit)  # Hit AXP
                else:
                    logger.debug("expl: %s", hset)
                    expls.append(hset)

                    if len(expls) != xnum:
                        hitman.block(hset)  # Block CXP
                    else:
                        break
        logger.debug("Chances of further enhancements: %d", possibility)
        assert itr == (len(expls) + len(duals) + 1), "Assertion Error: " + ",".join(
            map(str, [itr, len(expls), len(duals)])
        )
        return expls, duals


class Cached:
    SOLVER = "solver"


class Stat:
    cache_hit = {Cached.SOLVER: 0}
    cache_miss = {Cached.SOLVER: 0}

    def inc_cache_hit(flag: str):
        Stat.cache_hit[flag] += 1

    def inc_cache_miss(flag: str):
        Stat.cache_miss[flag] += 1
