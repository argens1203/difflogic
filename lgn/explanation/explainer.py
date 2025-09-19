import logging
from typing import Optional, Set

# if TYPE_CHECKING:
from lgn.encoding import Encoding
from experiment.helpers import (
    Context,
    Partial_Inp_Set,
    Transformed_Partial_Inp_Set,
)

from .multiclass_solver import MulticlassSolver
from .instance import Instance
from .session import Session

logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, encoding: Encoding, ctx: Context):
        self.encoding = encoding
        self.oracle = MulticlassSolver(encoding=encoding, ctx=ctx)

    def explain(self, instance: Instance):
        pred_class = instance.get_predicted_class()
        inp = instance.get_input()

        logger.debug("Explaining Input: %s", inp)

        logger.debug("Predicted Class - %s", pred_class)

        assert not self.oracle.is_solvable(
            pred_class=pred_class, inp=inp
        ), "Assertion Error: " + str(inp)

        axp = self.reduce_axp(inp, pred_class)
        logger.debug("One AXP: %s", axp)
        return axp

    def _enumerate_unit_mcs(session: Session):
        options = set(session.options)
        counter = 0
        for hypo in options:
            if session.is_solvable_with_opt(options - {hypo}):
                session.hit({hypo})  # MCS is registered here, stored in session.
                counter += 1

        return counter

    def _extract_mcs(
        session: Session,
        guess: Transformed_Partial_Inp_Set,
        model: Transformed_Partial_Inp_Set,
    ) -> Set[int]:
        # CXP lies within unmatching features (between inp and guess)
        inp = set(session.options)

        assert inp - guess - model == inp - model

        uncertain_set = inp - guess - model
        satsifiable_set = model & (inp)

        # Test each hypo in uncertain_set

        mcs = set()
        for hypo in uncertain_set:
            # Keep adding while satisfiable
            if session.is_solvable_with_opt(inp=satsifiable_set | {hypo}):
                satsifiable_set.add(hypo)
            else:
                # Partial MCS found in a reversed manner
                mcs.add(hypo)

        logger.debug("To hit: %s", mcs)
        # the entirity of to_hit is a MCS
        return mcs

    def mhs_mus_enumeration(
        self, instance: Instance, xnum: Optional[int] = None, smallest=False
    ):
        session: Session

        with Session.use_context(
            instance=instance,
            hit_type="sorted" if smallest else "lbx",
            oracle=self.oracle,
        ) as session:

            # Try unit-MCSes
            preempt_hit = Explainer._enumerate_unit_mcs(session)
            session.add_to_itr(preempt_hit)

            # Main Loop
            while True:
                # Get a guess
                guess = session.get()
                if guess == None:
                    break

                # Try the guess
                res = session.solve_opt(inp=guess)

                # If guess is MUS, block it
                if not res["solvable"]:
                    session.block(guess)
                    if xnum is not None and session.get_expls_count() >= xnum:
                        break
                    else:
                        continue

                # Else extract MCS from the guess
                mcs = Explainer._extract_mcs(session, guess=guess, model=res["model"])
                session.hit(mcs)

            # Extact outputs
            expls = session.get_expls_opt()
            duals = session.get_duals_opt()
            itr = session.get_itr()

            # Check itration count
            if xnum is None:
                assert itr == (
                    len(session.expls) + len(session.duals) + 1
                ), "Assertion Error: " + ",".join(
                    map(str, [itr, len(expls), len(duals)])
                )

            return expls, duals

    def _enumerate_unit_mus(session: Session):
        options = set(session.options)
        itr = 0

        for hypo in options:
            # Unit-size MUS
            if not session.is_solvable_with_opt(inp={hypo}):
                session.hit({hypo})
                itr += 1

            # Unit-size MCS
            if session.is_solvable_with_opt(inp=options - {hypo}):
                session.block({hypo})
                itr += 1

        return itr

    def mhs_mcs_enumeration(
        self,
        instance: Instance,
        xnum: Optional[int] = None,
        smallest=False,
    ):
        session: Session

        with Session.use_context(
            instance=instance,
            hit_type="sorted" if smallest else "lbx",
            oracle=self.oracle,
        ) as session:

            inp = set(session.options)
            counter = Explainer._enumerate_unit_mus(session=session)
            session.add_to_itr(counter)

            # Main Loop
            while True:
                # Get a guess
                hset = session.get()
                # print("guess", hset)
                if hset == None:
                    break

                # Try the guess
                res = session.solve_opt(inp=inp - hset)

                # If guess is MCS, block it
                if res["solvable"]:
                    session.block(hset)
                    if xnum is not None and session.get_expls_count() >= xnum:
                        break
                    else:
                        continue

                # Else extract MUS from the guess
                to_hit = self.reduce_axp_opt(
                    inp=res["core"],
                    session=session,
                )
                to_hit = set(to_hit)

                session.hit(to_hit)

            # Extract outputs
            itr = session.get_itr()
            expls = session.get_expls_opt()
            duals = session.get_duals_opt()

            if xnum is None:
                # Check iteration count
                assert itr == (
                    len(expls) + len(duals) + 1
                ), "Assertion Error: " + ",".join(
                    map(str, [itr, len(expls), len(duals)])
                )
            return expls, duals

    # PRIVATE

    def reduce_axp_opt(self, inp: Transformed_Partial_Inp_Set, session: Session):
        tmp_inp = inp.copy()
        for hypo in inp:
            if not session.is_solvable_with_opt(inp=tmp_inp - {hypo}):
                tmp_inp = tmp_inp - {hypo}

        return tmp_inp

    def reduce_axp(self, inp: Partial_Inp_Set, predicted_cls: int):
        """
        Get one AXP for the input and predicted class

        :param inp: input features
        :param predicted_cls: predicted class

        :return: AXP
        """
        tmp_input = inp.copy()
        for part in self.encoding.get_parts():
            logger.debug("Testing removal of input %s", part)
            tt_input = Explainer.remove_part(tmp_input, part)
            if not self.oracle.is_solvable(pred_class=predicted_cls, inp=tt_input):
                tmp_input = tt_input
            else:
                continue
        return tmp_input

    # NEW
    @staticmethod
    def remove_part(inp: set[int], part: list[int]) -> set[int]:
        to_remove = set()
        for i in part:
            to_remove.add(i)
            to_remove.add(-i)
        return inp - to_remove

    # NEW

    def explain_both_and_assert(self, instance, xnum: Optional[int]):
        self.explain(instance)

        axps, axp_dual = self.mhs_mus_enumeration(instance, xnum=xnum)
        cxps, cxp_dual = self.mhs_mcs_enumeration(instance, xnum=xnum)

        logger.debug("Input: %s", instance.get_input())
        logger.debug(
            "AXPs: %s",
            str([sorted(one) for one in sorted(axps, key=lambda x: (len(x), x[0]))]),
        )
        logger.debug(
            "Duals: %s",
            str(
                [sorted(one) for one in sorted(axp_dual, key=lambda x: (len(x), x[0]))]
            ),
        )
        logger.debug(
            "CXPs: %s",
            str([sorted(one) for one in sorted(cxps, key=lambda x: (len(x), x[0]))]),
        )
        logger.debug(
            "Duals: %s",
            str(
                [sorted(one) for one in sorted(cxp_dual, key=lambda x: (len(x), x[0]))]
            ),
        )
        axp_set = set()
        for axp in axps:
            axp_set.add(frozenset(axp))
        cxp_set = set()
        for cxp in cxps:
            cxp_set.add(frozenset(cxp))
        axp_dual_set = set()
        for axp_d in axp_dual:
            axp_dual_set.add(frozenset(axp_d))
        cxp_dual_set = set()
        for cxp_d in cxp_dual:
            cxp_dual_set.add(frozenset(cxp_d))

        if xnum is None:
            assert axp_set.difference(cxp_dual_set) == set()
            assert cxp_dual_set.difference(axp_set) == set()

            assert axp_dual_set.difference(cxp_set) == set()
            assert cxp_set.difference(axp_dual_set) == set()

        axps = [instance.verbose(axp) for axp in axps]
        cxps = [instance.verbose(cxp) for cxp in cxps]
        for i, axp in enumerate(axps):
            logger.debug("AXP #%d: %s", i, axp)
        for i, cxp in enumerate(cxps):
            logger.debug("CXP #%d: %s", i, cxp)
        logger.debug("\n")

        return len(axps) + len(axp_dual)
