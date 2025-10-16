import time
import logging

from tqdm import tqdm

from experiment.args.args import DefaultArgs
from experiment.args.pysat_args import PySatArgs
from lgn.explanation import Explainer, Instance
from experiment.args import ExplainerArgs
from .helpers import Context

logger = logging.getLogger(__name__)


class Explain:
    # ---- ---- ---- ---- ---- EXPLAINERS  ---- ---- ---- ---- ---- #
    @staticmethod
    def explain_raw(
        args: DefaultArgs,
        explainer: Explainer,
        encoding,
        ctx: Context,
        raw=None,
        inp=None,
    ) -> tuple[float, int, int]:
        start = time.time()
        ctx.logger.debug("Raw: %s\n", raw)
        ctx.logger.debug("Inp: %s\n", inp)
        instance = Instance.from_encoding(encoding=encoding, raw=raw, inp=inp)
        exp_count = explainer.explain_both_and_assert(
            instance, xnum=args.xnum, args=args
        )

        ctx.inc_num_explanations(exp_count)
        return time.time() - start, exp_count, 1

    @staticmethod
    def explain_one(
        args: DefaultArgs, explainer: Explainer, encoding, ctx: Context
    ) -> tuple[float, int, int]:
        start = time.time()
        exp_count = 0

        batch, label, idx = next(iter(ctx.test_loader))
        for feat, index in zip(batch, idx):

            raw = ctx.get_raw(index, is_train=False)
            ctx.logger.debug("Raw: %s\n", raw)

            instance = Instance.from_encoding(encoding=encoding, feat=feat)
            explainer.explain(instance)
            exp_count += 1
            break

        ctx.inc_num_explanations(exp_count)
        return time.time() - start, exp_count, 1

    @staticmethod
    def explain_all(
        args: DefaultArgs, explainer, encoding, ctx: Context
    ) -> tuple[float, int, int]:
        all_times = 0
        exp_count = 0
        count = 0

        remaining_time = args.max_explain_time
        for data_loader, is_train in zip(
            [ctx.train_loader, ctx.test_loader], [True, False]
        ):
            t, e, c = Explain.explain_dataloader(
                data_loader=data_loader,
                exp_args=ExplainerArgs(xnum=args.xnum, max_explain_time=remaining_time),
                is_train=is_train,
                explainer=explainer,
                encoding=encoding,
                ctx=ctx,
                pysat_args=args,
            )
            all_times += t
            exp_count += e
            count += c
            remaining_time -= t

        ctx.inc_num_explanations(exp_count)
        return all_times, exp_count, count

    @staticmethod
    def explain_dataloader(
        data_loader,
        exp_args: ExplainerArgs,
        explainer: Explainer,
        pysat_args: PySatArgs,
        encoding,
        ctx: Context,
        is_train=False,
    ) -> tuple[float, int, int]:
        all_times = 0
        exp_count = 0
        count = 0
        max_explain_time = exp_args.max_explain_time

        logging.info(
            "Explaining dataloader with max_explain_time: %s", max_explain_time
        )
        begin = time.time()
        batch_idx = 0
        for batch, label, idx in data_loader:
            start = time.time()
            logging.info("Explaining batch: %s", batch_idx + 1)
            for feat, i in tqdm(zip(batch, idx), total=len(batch)):
                raw = ctx.get_raw(i, is_train=is_train)
                ctx.logger.debug("Raw: %s\n", raw)

                instance = Instance.from_encoding(encoding=encoding, feat=feat)
                exp_count_axp_plus_cxp = explainer.explain_both_and_assert(
                    instance, xnum=exp_args.xnum, args=pysat_args
                )
                exp_count += exp_count_axp_plus_cxp
                count += 1
                if (
                    max_explain_time is not None
                    and time.time() - begin > max_explain_time
                ):
                    break
            if max_explain_time is not None and time.time() - begin > max_explain_time:
                break
            all_times += time.time() - start
            batch_idx += 1

        ctx.inc_num_explanations(exp_count)
        return all_times, exp_count, count
