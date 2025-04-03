import logging

from pysat.examples.hitman import Hitman

from lgn.encoding import Encoding
from lgn.util import input_to_feat, Stat
from lgn.util import Inp, Partial_Inp, Htype

from .multiclass_solver import MulticlassSolver
from .instance import Instance
from .session import Session

logger = logging.getLogger(__name__)


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
        logger.debug("Starting mhs_mus_enumeration")
        logger.debug("Input: %s", instance.get_input())
        """
        Enumerate subset- and cardinality-minimal explanations.
        """

        session: Session

        def enumerate_unit_mcs(session: Session, inp: list):  # Get unit-size MCSes
            preempt_hit = 0
            for i, hypo in enumerate(inp):

                remaining = inp[:i] + inp[(i + 1) :]
                if session.is_solvable_with(set(remaining)):
                    session.hit([hypo])  # Found unit-size MCS
                    preempt_hit += 1

            return preempt_hit

        with Session.use_context(
            instance=instance,
            hit_type="sorted" if smallest else "lbx",
            oracle=self.oracle,
        ) as session:
            inp = instance.get_input()

            # Try unit-MCSes
            preempt_hit = enumerate_unit_mcs(session, inp)
            session.add_to_itr(preempt_hit)

            # main loop
            while True:
                hset = session.get()  # Get candidate MUS
                if hset == None:  # Terminates when there is no more candidate MUS
                    break

                res = session.solve(inp=set(hset))
                model = res["model"]
                solvable = res["solvable"]

                if not solvable:
                    logger.debug("Is NOT satisfied %s", hset)

                    session.block(hset)
                    if session.get_expls_count() > xnum:
                        break
                else:
                    logger.debug("IS satisfied %s", hset)

                    # CXP lies within removed features
                    unsatisfied = set(inp) - set(hset) - set(model)
                    hset = set(model) & (set(inp))

                    logger.debug("Unsatisfied: %s", unsatisfied)
                    logger.debug("Hset: %s", hset)

                    to_hit = []
                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        if session.is_solvable_with(
                            inp=hset | {h}
                        ):  # Keep adding while satisfiable
                            hset.add(h)
                        else:
                            to_hit.append(h)
                            # Partial MCS found in a reversed manner

                    logger.debug("To hit: %s", to_hit)

                    session.hit(to_hit)  # the entirity of to_hit is a MCS

        expls = session.get_expls()
        duals = session.get_duals()
        itr = session.get_itr()

        assert itr == (
            len(session.expls) + len(session.duals) + 1
        ), "Assertion Error: " + ",".join(map(str, [itr, len(expls), len(duals)]))

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
            logger.debug("Starting mhs_mcs_enumeration")
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
                logger.debug("itr %s) cand: %s", itr, hset)

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

                    logger.debug("to_hit: %s", to_hit)

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
