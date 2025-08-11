import random
import logging
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
from lgn.encoding import Encoder, SatEncoder, BddEncoder, Validator
from lgn.explanation import Explainer, Instance
from lgn.dataset import (
    new_load_dataset as load_dataset,
)
from lgn.model import get_model, compile_model, train_eval, multi_eval
from .util import get_results, Stat, ExplainerArgs, get_enc_type
from constant import device

from constant import Stats
import time

torch.set_num_threads(1)  # ???


def seed_all(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Context:
    def __init__(self, args):
        self.logger = logging.getLogger()

        eid = args.experiment_id if args.experiment_id is not None else 0
        self.results = get_results(eid, args)
        self.train_loader, self.test_loader, self.get_raw, self.dataset = load_dataset(
            args
        )
        self.verbose = args.verbose


class OneExperiment:
    @staticmethod
    def get_ctx(args):
        return Context(args)

    @staticmethod
    def run_presentation(args):
        ctx = OneExperiment.get_ctx(args)

        model = OneExperiment.get_model(args, ctx=ctx)

        encoding = OneExperiment.get_encoding(
            model=model,
            enc_type=get_enc_type(args.enc_type),
            deduplication=args.deduplicate,
            ctx=ctx,
        )
        explainer = OneExperiment.get_explainer(encoding, ctx=ctx)

        if args.explain is not None:
            raw = args.explain.split(",")
            OneExperiment.explain_raw(raw, args, explainer, encoding, ctx=ctx)
        elif args.explain_all:
            OneExperiment.explain_all(args, explainer, encoding, ctx=ctx)
        elif args.explain_one:
            OneExperiment.explain_one(args, explainer, encoding, ctx=ctx)
        else:
            OneExperiment.explain_dataloader(
                ctx.test_loader,
                args,
                explainer=explainer,
                encoding=encoding,
                is_train=False,
                ctx=ctx,
            )

        return None

    @staticmethod
    def compare_encoders(args):
        ctx = OneExperiment.get_ctx(args)
        model = OneExperiment.get_model(args, ctx=ctx)

        encoding2 = OneExperiment.get_encoding(
            model=model,
            enc_type=get_enc_type(args.enc_type),
            deduplication="bdd",
            ctx=ctx,
        )
        encoding3 = OneExperiment.get_encoding(
            model=model,
            enc_type=get_enc_type(args.enc_type),
            deduplication="sat",
            ctx=ctx,
        )
        encoding1 = OneExperiment.get_encoding(
            model=model,
            enc_type=get_enc_type(args.enc_type),
            deduplication=None,
            ctx=ctx,
        )

        Validator.validate_encodings_with_data(
            encoding1=encoding1, encoding2=encoding2, dataloader=ctx.test_loader
        )
        Validator.validate_encodings_with_data(
            encoding1=encoding1, encoding2=encoding3, dataloader=ctx.test_loader
        )
        Validator.validate_encodings_with_data(
            encoding1=encoding2, encoding2=encoding3, dataloader=ctx.test_loader
        )

        Validator.validate_encodings_with_truth_table(
            encoding1=encoding1, encoding2=encoding2, dataset=ctx.dataset
        )
        Validator.validate_encodings_with_truth_table(
            encoding1=encoding1, encoding2=encoding3, dataset=ctx.dataset
        )
        Validator.validate_encodings_with_truth_table(
            encoding1=encoding2, encoding2=encoding3, dataset=ctx.dataset
        )

        input("All encodings are valid. Press Enter to continue...")

        encoding2.print()
        encoding3.print()

    @staticmethod
    def run_experiment(args):
        ctx = Context(args)
        # Asserts that results is not None, and enforces that entire test_set is explained
        model = OneExperiment.get_model(args, ctx=ctx)

        Stat.start_memory_usage()

        encoding = OneExperiment.get_encoding(
            model=model,
            enc_type=get_enc_type(args.enc_type),
            deduplication=args.deduplicate,
            ctx=ctx,
        )

        # Validator.validate_with_truth_table(encoding=self.encoding, model=self.model)
        encoding.print()
        explainer = OneExperiment.get_explainer(encoding, ctx=ctx)

        total_time_taken, exp_count, count = OneExperiment.explain_dataloader(
            ctx.test_loader,
            args,
            explainer=explainer,
            encoding=encoding,
            is_train=False,
            ctx=ctx,
        )
        # ============= ============= ============= ============= ============= ============= ============= =============

        ctx.results.store_explanation_stat(exp_count / count, Stats["deduplication"])
        ctx.results.store_resource_usage(
            total_time_taken / exp_count, Stat.get_memory_usage()
        )
        ctx.results.store_counts(count, exp_count)
        Stat.end_memory_usage()
        ctx.results.save()

        return ctx.results

    @staticmethod
    def find_model(args, ctx):
        model = OneExperiment.get_model(args, ctx=ctx)
        ctx.results.save()
        return ctx.results, model

    # ---- ---- ---- ---- ---- EXPLAINERS  ---- ---- ---- ---- ---- #
    @staticmethod
    def explain_raw(raw, args, explainer, encoding, ctx):
        start = time.time()
        ctx.logger.info("Raw: %s\n", raw)
        instance = Instance.from_encoding(encoding=encoding, raw=raw)
        exp_count = explainer.explain_both_and_assert(instance, xnum=args.xnum)
        return time.time() - start, exp_count, 1

    @staticmethod
    def explain_one(args, explainer, encoding, ctx):
        start = time.time()

        batch, label, idx = next(iter(ctx.test_loader))
        for feat, index in zip(batch, idx):

            raw = ctx.get_raw(index, is_train=False)
            ctx.logger.info("Raw: %s\n", raw)

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
            t, e, c = OneExperiment.explain_dataloader(
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
                ctx.logger.info("Raw: %s\n", raw)

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

    # ---- ---- ---- ---- ---- GETTERS ---- ---- ---- ---- ---- #

    @staticmethod
    def train_model(args, model, loss_fn, optim, ctx):
        train_eval(
            args,
            ctx.train_loader,
            None,
            ctx.test_loader,
            model,
            loss_fn,
            optim,
            ctx.results,
        )

    @staticmethod
    def eval_model(args, model, ctx):
        multi_eval(
            model,
            ctx.train_loader,
            ctx.test_loader,
            None,
            results=ctx.results,
            packbits_eval=args.packbits_eval,
        )

    @staticmethod
    def get_model(args, ctx):
        model, loss_fn, optim = get_model(args, ctx.results)
        if args.save_model and args.load_model:
            try:
                model.load_state_dict(
                    torch.load(args.model_path, map_location=torch.device(device))
                )
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                OneExperiment.train_model(args, model, loss_fn, optim, ctx)
                OneExperiment.eval_model(args, model, ctx)
                torch.save(model.state_dict(), args.model_path)

        elif args.save_model:
            OneExperiment.train_model(args, model, loss_fn, optim, ctx)
            OneExperiment.eval_model(args, model, ctx)
            torch.save(model.state_dict(), args.model_path)

        elif args.load_model:
            model.load_state_dict(
                torch.load(args.model_path, map_location=torch.device(device))
            )

        else:
            OneExperiment.train_model(args, model, loss_fn, optim, ctx)
            OneExperiment.eval_model(args, model, ctx)

        ####################################################################################################################
        if ctx.results is not None:
            ctx.results.store_custom("model_complete_time", time.time())

        if args.compile_model:
            compile_model(args, model, ctx.test_loader)

        return model

    @staticmethod
    def get_encoding(model, enc_type, ctx, deduplication: Optional[str] = None):
        _Encoder = Encoder
        if deduplication == "sat":
            _Encoder = SatEncoder
        elif deduplication == "bdd":
            _Encoder = BddEncoder

        encoding = _Encoder().get_encoding(
            model,
            ctx.dataset,
            enc_type=enc_type,
        )

        if ctx.results is not None:
            ctx.results.store_encoding(encoding)
        if ctx.verbose:
            encoding.print()

        return encoding

    @staticmethod
    def get_explainer(encoding, ctx):
        ctx.explainer = Explainer(encoding)
        return ctx.explainer
