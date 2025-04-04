import logging
from typing import Set

from pysat.examples.hitman import Hitman

from lgn.encoding import Encoding
from lgn.util import input_to_feat, Stat
from lgn.util import Inp, Partial_Inp, Htype, Partial_Inp_Set, Inp_Set

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
        inp = list(instance.get_input())

        logger.info("\n")
        logger.info("Explaining Input: %s", inp)

        logger.debug("Predicted Class - %s", pred_class)

        assert not self.oracle.is_solvable(
            pred_class=pred_class, inp=inp
        ), "Assertion Error: " + ",".join(map(str, inp))

        axp = self.reduce_axp(inp, pred_class)
        logger.info("One AXP: %s", axp)
        return axp

    def _enumerate_unit_mcs(session: Session, inp: Inp_Set):
        counter = 0
        for hypo in inp:
            if session.is_solvable_with(inp - {hypo}):
                session.hit({hypo})  # MCS is registered here, stored in session.
                counter += 1

        return counter

    def _extract_mcs(
        session: Session, inp: Inp_Set, guess: Partial_Inp_Set, model: Inp_Set
    ) -> Set[int]:
        # CXP lies within unmatching features (between inp and guess)
        uncertain_set = inp - guess - model
        satsifiable_set = model & (inp)

        # Test each hypo in uncertain_set

        mcs = set()
        for hypo in uncertain_set:
            # Keep adding while satisfiable
            if session.is_solvable_with(inp=satsifiable_set | {hypo}):
                satsifiable_set.add(hypo)
            else:
                # Partial MCS found in a reversed manner
                mcs.add(hypo)

        logger.debug("To hit: %s", mcs)
        # the entirity of to_hit is a MCS
        return mcs

    def mhs_mus_enumeration(self, instance: Instance, xnum=1000, smallest=False):
        session: Session
        inp = instance.get_input()

        with Session.use_context(
            instance=instance,
            hit_type="sorted" if smallest else "lbx",
            oracle=self.oracle,
        ) as session:

            # Try unit-MCSes
            preempt_hit = Explainer._enumerate_unit_mcs(session, inp)
            session.add_to_itr(preempt_hit)

            # Main Loop
            while True:
                # Get a guess
                guess = session.get()
                if guess == None:
                    break

                # Try the guess
                res = session.solve(inp=guess)

                # If guess is MUS, block it
                if not res["solvable"]:
                    session.block(guess)
                    if session.get_expls_count() > xnum:
                        break
                    else:
                        continue

                # Else extract MCS from the guess
                mcs = Explainer._extract_mcs(
                    session, inp=inp, guess=guess, model=res["model"]
                )
                session.hit(mcs)

            # Extact outputs
            expls = session.get_expls()
            duals = session.get_duals()
            itr = session.get_itr()

            # Check itration count
            assert itr == (
                len(session.expls) + len(session.duals) + 1
            ), "Assertion Error: " + ",".join(map(str, [itr, len(expls), len(duals)]))

            return expls, duals

    def _enumerate_unit_mus(session: Session, inp: Inp_Set):
        itr = 0

        for hypo in inp:
            # Unit-size MUS
            if not session.is_solvable_with(inp={hypo}):
                session.hit({hypo})
                itr += 1

            # Unit-size MCS
            if session.is_solvable_with(inp=inp - {hypo}):
                session.block({hypo})
                itr += 1

        return itr

    def mhs_mcs_enumeration(
        self,
        instance: Instance,
        xnum=1000,
        smallest=False,
    ):
        inp = instance.get_input()
        session: Session

        with Session.use_context(
            instance=instance,
            hit_type="sorted" if smallest else "lbx",
            oracle=self.oracle,
        ) as session:

            counter = Explainer._enumerate_unit_mus(session=session, inp=inp)
            session.add_to_itr(counter)

            # Main Loop
            while True:
                # Get a guess
                hset = session.get()
                if hset == None:
                    break

                # Try the guess
                res = session.solve(inp=inp - hset)

                # If guess is MCS, block it
                if res["solvable"]:
                    session.block(hset)
                    if session.get_expls_count() >= xnum:
                        break
                    else:
                        continue

                # Else extract MUS from the guess
                to_hit = self.reduce_axp(
                    inp=list(res["core"]),
                    predicted_cls=instance.get_predicted_class(),
                )
                to_hit = set(to_hit)

                session.hit(to_hit)

            # Extract outputs
            itr = session.get_itr()
            expls = session.get_expls()
            duals = session.get_duals()

            # Check iteration count
            assert itr == (len(expls) + len(duals) + 1), "Assertion Error: " + ",".join(
                map(str, [itr, len(expls), len(duals)])
            )
            return expls, duals

    # PRIVATE

    def reduce_axp(self, inp, predicted_cls):
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
