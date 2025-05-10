import random
import logging
import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
from lgn.encoding import Encoding
from lgn.explanation import Explainer, Instance
from lgn.dataset import (
    get_dataset,
    new_load_dataset as load_dataset,
)
from lgn.model import get_model, compile_model, train_eval, multi_eval
from lgn.util import get_args, get_results, setup_logger, Stat
from constant import Args
from pysat.card import EncType

from constant import Stats
import time

torch.set_num_threads(1)  # ???


def seed_all(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


from lgn.util import DefaultArgs, ExplainerArgs
from .settings import Settings
from .util import get_enc_type


class OneExperiment:
    def __init__(self, args):
        self.logger = logging.getLogger()
        if args.deduplicate:
            Args["Deduplicate"] = True  # TODO: find ways to store global args

        self.results = get_results(args.experiment_id, args)
        self.train_loader, self.test_loader, self.get_raw, self.dataset = load_dataset(
            args
        )
        self.verbose = args.verbose

    def run_presentation(self, args):
        if args.deduplicate:
            Args["Deduplicate"] = True  # TODO: find ways to store global args

        self.get_model(args)
        self.get_encoding(enc_type=get_enc_type(args.enc_type))
        self.get_explainer()

        if args.explain is not None:
            raw = args.explain.split(",")
            self.explain_raw(raw, args)
        elif args.explain_all:
            self.explain_all(args)
        elif args.explain_one:
            self.explain_one(args)
        else:
            self.explain_dataloader(self.test_loader, args, is_train=False)

        return None

    def run_experiment(self, args):
        # Asserts that results is not None, and enforces that entire test_set is explained
        if args.deduplicate:
            Args["Deduplicate"] = True  # TODO: find ways to store global args

        self.get_model(args)

        Stat.start_memory_usage()

        self.get_encoding(enc_type=get_enc_type(args.enc_type))
        self.get_explainer()

        total_time_taken, exp_count, count = self.explain_dataloader(
            self.test_loader, args, is_train=False
        )
        # ============= ============= ============= ============= ============= ============= ============= =============

        self.results.store_explanation_stat(exp_count / count, Stats["deduplication"])
        self.results.store_resource_usage(
            total_time_taken / count, Stat.get_memory_usage()
        )
        self.results.save()

        return self.results

    # ---- ---- ---- ---- ---- EXPLAINERS  ---- ---- ---- ---- ---- #

    def explain_raw(self, raw, args):
        start = time.time()
        self.logger.info("Raw: %s\n", raw)
        instance = Instance.from_encoding(encoding=self.encoding, raw=raw)
        exp_count = self.explainer.explain_both_and_assert(instance, xnum=args.xnum)
        return time.time() - start, exp_count, 1

    def explain_one(self, args):
        start = time.time()

        batch, label, idx = next(iter(self.test_loader))
        for feat, index in zip(batch, idx):

            raw = self.get_raw(index, is_train=False)
            self.logger.info("Raw: %s\n", raw)

            instance = Instance.from_encoding(encoding=self.encoding, feat=feat)
            exp_count = self.explainer.explain_both_and_assert(instance, xnum=args.xnum)
            return time.time() - start, exp_count, 1

    def explain_all(self, args):
        all_times = 0
        exp_count = 0
        count = 0

        for data_loader, is_train in zip(
            [self.train_loader, self.test_loader], [True, False]
        ):
            t, e, c = self.explain_dataloader(
                data_loader=data_loader,
                exp_args=ExplainerArgs(xnum=args.xnum),
                is_train=is_train,
            )
            all_times += t
            exp_count += e
            count += c

        return all_times, exp_count, count

    def explain_dataloader(
        self,
        data_loader,
        exp_args: ExplainerArgs,
        is_train=False,
    ):
        all_times = 0
        exp_count = 0
        count = 0

        for batch, label, idx in tqdm(data_loader):
            start = time.time()
            for feat, i in tqdm(zip(batch, idx)):
                raw = self.get_raw(i, is_train=is_train)
                self.logger.info("Raw: %s\n", raw)

                instance = Instance.from_encoding(encoding=self.encoding, feat=feat)
                exp_count_axp_plus_cxp = self.explainer.explain_both_and_assert(
                    instance, xnum=exp_args.xnum
                )
                exp_count += exp_count_axp_plus_cxp
            all_times += time.time() - start
            count += len(batch)

        return all_times, exp_count, count

    # ---- ---- ---- ---- ---- GETTERS ---- ---- ---- ---- ---- #

    def get_model(self, args):
        model, loss_fn, optim = get_model(args, self.results)

        if not args.save_model and not args.load_model:
            try:
                model.load_state_dict(torch.load(args.model_path))
            except Exception as e:
                print(e)
                train_eval(
                    args,
                    self.train_loader,
                    None,
                    self.test_loader,
                    model,
                    loss_fn,
                    optim,
                    self.results,
                )
                torch.save(model.state_dict(), args.model_path)
            multi_eval(
                model,
                self.train_loader,
                self.test_loader,
                None,
                results=self.results,
                packbits_eval=args.packbits_eval,
            )

        if args.load_model:
            print("Loading Model...")
            model.load_state_dict(torch.load(args.model_path))

        ####################################################################################################################
        if self.results is not None:
            self.results.store_custom("model_complete_time", time.time())

        if args.save_model:
            torch.save(model.state_dict(), args.model_path)

        if args.compile_model:
            compile_model(args, model, self.test_loader)

        self.model = model

    def get_encoding(self, enc_type):
        self.encoding = Encoding(self.model, self.dataset, enc_type=enc_type)
        # Validator.validate_with_truth_table(encoding=encoding, model=model)

        if self.results is not None:
            self.results.store_encoding(self.encoding)

        if self.verbose:
            self.encoding.print()
            self.logger.info("\n")

    def get_explainer(self):
        self.explainer = Explainer(self.encoding)
