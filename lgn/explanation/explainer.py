import logging
import math
from typing import Optional, Set

# if TYPE_CHECKING:
from experiment.args.pysat_args import PySatArgs
from experiment.args.explainer_args import ExplainerArgs
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
        for cls in self.encoding.get_classes():
            assert self.oracle.is_solvable(pred_class=cls, inp=set())
        self.ctx = ctx

    def explain(self, instance: Instance):
        # print(instance.feat)
        # print(instance.get_input())
        pred_class = instance.get_predicted_class()
        inp = instance.get_input()

        logger.debug("Explaining Input: %s", inp)

        logger.debug("Predicted Class - %s", pred_class)

        assert not self.oracle.is_solvable(
            pred_class=pred_class, inp=inp
        ), "Assertion Error: " + str(inp)

        axp = self.reduce_axp(inp, pred_class)
        logger.debug("One AXP: %s", axp)
        self.ctx.record_solving_stats(
            self.oracle.get_clause_count(), self.oracle.get_var_count()
        )
        return axp

    @staticmethod
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
        self, instance: Instance, args: PySatArgs, xnum: Optional[int] = None
    ):
        session: Session

        with Session.use_context(
            instance=instance,
            hit_type=args.h_type,
            oracle=self.oracle,
            solver=args.h_solver,
            e_ctx=self.ctx,
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

            itr = session.get_itr()
            # Check itration count
            if xnum is None:
                assert itr == (
                    len(session.expls) + len(session.duals) + 1
                ), "Assertion Error: " + ",".join(
                    map(str, [itr, len(session.expls), len(session.duals)])
                )

                # Also if dual is None, it means the entire input is unSAT
                if len(session.duals) == 0:
                    session.hit(set(session.options))

            # Extact outputs
            expls = session.get_expls_opt()
            duals = session.get_duals_opt()

            return expls, duals

    @staticmethod
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
        args: PySatArgs,
        xnum: Optional[int] = None,
    ):
        session: Session

        with Session.use_context(
            instance=instance,
            hit_type=args.h_type,
            solver=args.h_solver,
            oracle=self.oracle,
            e_ctx=self.ctx,
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

            if xnum is None:
                # Check iteration count
                assert itr == (
                    len(session.expls) + len(session.duals) + 1
                ), "Assertion Error: " + ",".join(
                    map(str, [itr, len(session.expls), len(session.duals)])
                )

                # Also if dual is None, it means the entire input is unSAT
                if len(session.duals) == 0:
                    session.hit(set(session.options))

            expls = session.get_expls_opt()
            duals = session.get_duals_opt()
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

    def combined_explain(
        self,
        instance,
        xnum: Optional[int],
        exp_args: ExplainerArgs,
        pysat_args: PySatArgs,
    ) -> int:
        if exp_args.explain_algorithm == "mus":
            axps, axp_dual = self.mhs_mus_enumeration(
                instance, xnum=xnum, args=pysat_args
            )
            return len(axps) + len(axp_dual)
        elif exp_args.explain_algorithm == "mcs":
            cxps, cxp_dual = self.mhs_mcs_enumeration(
                instance, xnum=xnum, args=pysat_args
            )
            return len(cxps) + len(cxp_dual)
        elif exp_args.explain_algorithm == "both":
            return self.explain_both_and_assert(instance, xnum=xnum, args=pysat_args)
        elif exp_args.explain_algorithm == "var":
            return self.variable_enumeration(
                instance, args=pysat_args, exp_args=exp_args
            )
        elif exp_args.explain_algorithm == "find_one":
            self.explain(instance)
            return 1
        else:
            raise ValueError(
                f"Unknown explanation algorithm: {exp_args.explain_algorithm}"
            )

    def explain_both_and_assert(self, instance, xnum: Optional[int], args: PySatArgs):
        self.explain(instance)

        axps, axp_dual = self.mhs_mus_enumeration(instance, xnum=xnum, args=args)
        cxps, cxp_dual = self.mhs_mcs_enumeration(instance, xnum=xnum, args=args)

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

        # print("cxp_dual_set:", cxp_dual_set)
        # print("cxp_set:", cxp_set)
        # print("axp_dual_set:", axp_dual_set)
        # print("axp_set:", axp_set)

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

        # input("Press Enter to continue...")
        return len(axps) + len(axp_dual)

    def variable_enumeration(self, instance, args: PySatArgs, exp_args: ExplainerArgs):
        # We start with MCS enumeration first
        last_time = math.inf
        session: Session
        with Session.use_context(
            instance=instance,
            hit_type=args.h_type,
            solver=args.h_solver,
            oracle=self.oracle,
            e_ctx=self.ctx,
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
                    continue

                # if res["core"] is None:
                #     print("No core found, breaking loop.")
                #     print("res:", res)
                #     print("inp:", inp)
                #     input("Press Enter to continue...")
                # Else extract MUS from the guess
                to_hit = self.reduce_axp_opt(
                    inp=res["core"],
                    session=session,
                )
                to_hit = set(to_hit)

                session.hit(to_hit)

                # if time_taken > last_time * (1 + args.switch_threshold):
                if (
                    session.get_duals_count() >= exp_args.explain_switch_window
                    and session.get_expls_count() > exp_args.explain_switch_window
                ):
                    recent_cxp_sizes = [
                        len(cxp)
                        for cxp in session.get_expls()[
                            -exp_args.explain_switch_window :
                        ]
                    ]
                    recent_axp_sizes = [
                        len(axp)
                        for axp in session.get_duals()[
                            -exp_args.explain_switch_window :
                        ]
                    ]
                    avg_cxp_size = sum(recent_cxp_sizes) / len(recent_cxp_sizes)
                    avg_axp_size = sum(recent_axp_sizes) / len(recent_axp_sizes)

                    # Switch when AXP >> CXP
                    if avg_axp_size / avg_cxp_size > exp_args.explain_switch_alpha:
                        logger.debug("Switching to MHS-MUS enumeration.")
                        break
            # Extract outputs
            # Note: this is in OHE space, not original feature space
            # eg.: cxps = [[1], [2]] means that both feature 1 and feature 2 are CXPs
            # It would translate to [1, -2] or [3, -4] or [-1, 2] depending on instance
            cxps = session.get_expls()
            axps = session.get_duals()

        with Session.use_context(
            instance=instance,
            hit_type=args.h_type,
            oracle=self.oracle,
            solver=args.h_solver,
            e_ctx=self.ctx,
        ) as session:

            # Commented out since it is already done in previous session
            # preempt_hit = Explainer._enumerate_unit_mcs(session)
            for c in cxps:
                session.hit(set(c))
                assert session.is_solvable_with_opt(set(session.options) - set(c))
            for a in axps:
                session.block(set(a))
                assert not session.is_solvable_with_opt(set(a))
            session.add_to_itr(len(cxps) + len(axps))
            # session.add_to_itr(preempt_hit)

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
                    continue

                # Else extract MCS from the guess
                mcs = Explainer._extract_mcs(session, guess=guess, model=res["model"])
                session.hit(mcs)

            # Extact outputs
            expls = session.get_expls_opt()
            duals = session.get_duals_opt()

            itr = session.get_itr()
            # input("Press Enter to continue...")

            return len(expls) + len(duals)
