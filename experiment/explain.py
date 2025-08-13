import time
from tqdm import tqdm

from lgn.explanation import Explainer, Instance
from experiment.args import ExplainerArgs


class Explain:
    # ---- ---- ---- ---- ---- EXPLAINERS  ---- ---- ---- ---- ---- #
    @staticmethod
    def explain_raw(raw, args, explainer, encoding, ctx):
        start = time.time()
        ctx.logger.debug("Raw: %s\n", raw)
        instance = Instance.from_encoding(encoding=encoding, raw=raw)
        exp_count = explainer.explain_both_and_assert(instance, xnum=args.xnum)
        return time.time() - start, exp_count, 1

    @staticmethod
    def explain_one(args, explainer, encoding, ctx):
        start = time.time()

        batch, label, idx = next(iter(ctx.test_loader))
        for feat, index in zip(batch, idx):

            raw = ctx.get_raw(index, is_train=False)
            ctx.logger.debug("Raw: %s\n", raw)

            instance = Instance.from_encoding(encoding=encoding, feat=feat)
            exp_count = explainer.explain_both_and_assert(instance, xnum=args.xnum)
            return time.time() - start, exp_count, 1

    @staticmethod
    def explain_all(args, explainer, encoding, ctx):
        all_times = 0
        exp_count = 0
        count = 0

        remaining_time = args.max_time
        for data_loader, is_train in zip(
            [ctx.train_loader, ctx.test_loader], [True, False]
        ):
            t, e, c = Explain.explain_dataloader(
                data_loader=data_loader,
                exp_args=ExplainerArgs(xnum=args.xnum, max_time=remaining_time),
                is_train=is_train,
                explainer=explainer,
                encoding=encoding,
                ctx=ctx,
            )
            all_times += t
            exp_count += e
            count += c
            remaining_time -= t

        return all_times, exp_count, count

    @staticmethod
    def explain_dataloader(
        data_loader,
        exp_args: ExplainerArgs,
        explainer: Explainer,
        encoding,
        ctx,
        is_train=False,
    ):
        all_times = 0
        exp_count = 0
        count = 0
        max_time = exp_args.max_time

        begin = time.time()
        for batch, label, idx in tqdm(data_loader):
            start = time.time()
            for feat, i in tqdm(zip(batch, idx)):
                raw = ctx.get_raw(i, is_train=is_train)
                ctx.logger.debug("Raw: %s\n", raw)

                instance = Instance.from_encoding(encoding=encoding, feat=feat)
                exp_count_axp_plus_cxp = explainer.explain_both_and_assert(
                    instance, xnum=exp_args.xnum
                )
                exp_count += exp_count_axp_plus_cxp
                if max_time is not None and time.time() - begin > max_time:
                    break
            if max_time is not None and time.time() - begin > max_time:
                break
            all_times += time.time() - start
            count += len(batch)

        return all_times, exp_count, count
