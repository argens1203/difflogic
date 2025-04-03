import logging
from contextlib import contextmanager

from pysat.examples.hitman import Hitman

from lgn.encoding import Encoding
from lgn.util import feat_to_input, input_to_feat, Stat

from .multiclass_solver import MulticlassSolver
from .instance import Instance

logger = logging.getLogger(__name__)

from typing import List

Inp = List[int]
Partial_Inp = List[int]
Htype = str  # "sorted" or "lbx"


class Session:
    def __init__(self, instance: Instance, hitman: Hitman, oracle: MulticlassSolver):
        self.instance = instance
        self.hitman = hitman
        self.duals = []
        self.expls = []
        self.oracle = oracle
        pass

    def is_solvable_with(self, inp: Partial_Inp):
        return self.oracle.is_solvable(self.instance.get_predicted_class(), inp=inp)

    def hit(self, hypo: Partial_Inp):
        self.hitman.hit(hypo)
        self.duals.append(hypo)

    def block(self, hypo: Partial_Inp):
        self.hitman.block(hypo)
        self.expls.append(hypo)
        pass

    def get(self):
        return self.hitman.get()

    @contextmanager
    def use_context(instance: Instance, htype: Htype = "lbx"):
        try:
            hitman = Hitman(
                bootstrap_with=[instance.get_input()],
                htype=htype,
            )
            yield Session(instance, hitman=hitman)
        finally:
            hitman.delete()  # Cleanup


class Explainer:
    def __init__(self, encoding: Encoding):
        self.encoding = encoding
        self.oracle = MulticlassSolver(encoding=encoding)

    def explain(self, instance):
        pred_class = instance.get_predicted_class()
        inp = instance.get_input()

        logger.info("\n")
        logger.info("Explaining Input: %s", inp)

        logger.debug("Predicted Class - %s", pred_class)

        assert not self.oracle.is_solvable(
            pred_class=pred_class, inp=inp
        ), "Assertion Error: " + ",".join(map(str, inp))

        axp = self.get_one_axp(inp, pred_class)
        logger.info("One AXP: %s", axp)
        return axp

    def mhs_mus_enumeration(self, instance, xnum=1000, smallest=False):
        """
        Enumerate subset- and cardinality-minimal explanations.
        """
        logger.debug("Starting mhs_mus_enumeration")

        inp = instance.get_input()
        pred_class = instance.get_predicted_class()

        expls = []
        duals = []
        htype = "sorted" if smallest else "lbx"

        with Hitman(bootstrap_with=[inp], htype=htype) as hitman:
            logger.info("Starting mhs_mus_enumeration")
            logger.info("Input: %s", inp)
            itr = 0
            # computing unit-size MCSes
            for i, hypo in enumerate(inp):
                if self.oracle.is_solvable(pred_class, inp=inp[:i] + inp[(i + 1) :]):
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

                res = self.oracle.solve(pred_class, inp=hset)
                model = res["model"]
                is_satisfiable = res["solvable"]

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
                        if self.oracle.is_solvable(
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

    def mhs_mcs_enumeration(
        self,
        instance: Instance,
        xnum=1000,
        smallest=False,
        reduce_="none",
        unit_mcs=False,
    ):
        """
        Enumerate subset- and cardinality-minimal contrastive explanations.
        """
        expls = []  # result
        duals = []  # just in case, let's save dual (abductive) explanations
        inp = instance.get_input()

        class_label = self.encoding.as_model()(input_to_feat(inp).reshape(1, -1)).item()
        pred_class = class_label + 1

        with Hitman(
            bootstrap_with=[inp], htype="sorted" if smallest else "lbx"
        ) as hitman:
            itr = 0
            logger.info("Starting mhs_mcs_enumeration")
            # computing unit-size MUSes
            for i, hypo in enumerate(inp):
                if not self.oracle.is_solvable(pred_class=pred_class, inp=[hypo]):
                    itr += 1
                    hitman.hit([hypo])
                    duals.append([hypo])
                elif unit_mcs and self.oracle.is_solvable(
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

                res = self.oracle.solve(
                    pred_class=pred_class,
                    inp=sorted(set(inp).difference(set(hset))),
                )
                solvable, core = res["solvable"], res["core"]
                if not solvable:
                    to_hit = core  # Core is a weak (non-minimal) AXP?

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
        assert itr == (len(expls) + len(duals) + 1), "Assertion Error: " + ",".join(
            map(str, [itr, len(expls), len(duals)])
        )
        return expls, duals

    # PRIVATE

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
            if not self.oracle.is_solvable(pred_class=predicted_cls, inp=tmp_input):
                continue
            else:
                tmp_input.append(feature)
        return tmp_input

    def __del__(self):
        logger.debug("Cache Hit: %s", str(Stat.cache_hit))
        logger.debug("Cache Miss: %s", str(Stat.cache_miss))
