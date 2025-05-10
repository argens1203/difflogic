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


class Settings:
    # Model training hyper parameters (according to the paper)
    all_temperatures = [1, 1 / 0.3, 1 / 0.1, 1 / 0.03, 1 / 0.01]
    learning_rate = 0.01
    epoch = 200
    batch_size = 100

    # Dataset specific hyper parameters (according to the paper)
    network_param = {
        "adult": {
            "num_neurons": 256,
            "num_layers": 5,
            "num_iterations": 200,
            "batch_size": 100,
            #        "temperature": None,
        },
        "monk1": {
            "num_neurons": 24,
            "num_layers": 6,
            "num_iterations": 200,
            "batch_size": 100,
        },
        "monk2": {
            "num_neurons": 12,
            "num_layers": 6,
            "num_iterations": 200,
            "batch_size": 100,
        },
        "monk3": {
            "num_neurons": 12,
            "num_layers": 6,
            "num_iterations": 200,
            "batch_size": 100,
        },
        "breast_cancer": {
            "num_neurons": 128,
            "num_layers": 5,
            "num_iterations": 200,
            "batch_size": 100,
        },
        "mnist": {
            "small": {
                "num_neurons": 8000,
                "num_layers": 6,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "normal": {
                "num_neurons": 64000,
                "num_layers": 6,
                "num_iterations": 200,
                "batch_size": 100,
            },
        },
        "cifar10": {
            "small": {
                "num_neurons": 12000,
                "num_layers": 4,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "medium": {
                "num_neurons": 128000,
                "num_layers": 4,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "large": {
                "num_neurons": 256000,
                "num_layers": 5,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "largeX2": {
                "num_neurons": 512000,
                "num_layers": 5,
                "num_iterations": 200,
                "batch_size": 100,
            },
            "largeX4": {
                "num_neurons": 1024000,
                "num_layers": 5,
                "num_iterations": 200,
                "batch_size": 100,
            },
        },
    }

    # Our setup for debugging
    dataset_neurons = [
        ("iris", 12),
        ("caltech101", 41 * 101),
        ("adult", 58),
        ("monk1", 10),
        ("monk2", 10),
        ("monk3", 10),
        ("breast_cancer", 26),
        ("mnist", 400),
    ]
    dataset_params = [
        (
            ds,
            {
                **{"num_neurons": neurons},
                **{
                    "num_layers": 2,
                    "num_iterations": 2000,
                    "batch_size": 100,
                },
            },
        )
        for ds, neurons in dataset_neurons
    ]
    debug_network_param = dict(dataset_params)

    def get_settings(dataset_name=None, paper=False, minimal=True):
        if not paper:
            return Settings.debug_network_param.get(dataset_name)
        if minimal:
            if dataset_name not in ["mnist", "cifar10"]:
                return Settings.network_param.get(dataset_name)
            else:
                return Settings.network_param.get(dataset_name).get("small")


default_args = {
    "eid": None,
    "dataset": "iris",
    "tau": 10,
    "seed": 0,
    "batch_size": 128,
    "learning_rate": 0.01,
    "training_bit_count": 32,
    "implementation": "cuda",
    "packbits_eval": False,
    "compile_model": False,
    "num_iterations": 100_000,
    "eval_freq": 2000,
    "valid_set_size": 0.0,
    "extensive_eval": False,
    "connections": "unique",
    "num_neurons": None,
    "num_layers": None,
    "grad_factor": 1.0,
    "get_formula": False,
    "verbose": False,
    "model_path": "model.pth",
    "save_model": False,
    "load_model": False,
    "explain": None,
    "explain_all": False,
    "explain_one": False,
    "xnum": 1000,
    "enc_type": "tot",
    "deduplicate": False,
    "experiment_id": None,
}


class Experiment:
    @staticmethod
    def get_enc_type(enc_type):
        return {
            "pw": EncType.pairwise,
            "seqc": EncType.seqcounter,
            "cardn": EncType.cardnetwrk,
            "sortn": EncType.sortnetwrk,
            "tot": EncType.totalizer,
            "mtot": EncType.mtotalizer,
            "kmtot": EncType.kmtotalizer,
        }[enc_type]

    def __init__(self):
        pass

    def debug(self, dataset=None):
        dataset = dataset if dataset is not None else "iris"
        dataset_args = Settings.debug_network_param.get(dataset)
        exp_args = {
            "eval_freq": 1000,
            "get_formula": True,
            "model_path": dataset + "_" + "model.pth",
            "verbose": True,
        }
        args = {**default_args, **exp_args, **dataset_args, **{"dataset": dataset}}
        self.run(args)

    def run_with_cmd(self):
        args = get_args()
        dataset_args = Settings.debug_network_param.get(args.dataset)
        exp_args = {
            "eval_freq": 1000,
            "get_formula": True,
            "model_path": args.dataset + "_" + "model.pth",
            "verbose": True,
        }
        args = {**vars(args), **default_args, **exp_args, **dataset_args}
        self.run(args)

    def f(self):
        experiment_id = 1
        all_res = []

        for datasets in ["monk1", "monk2", "monk3"]:
            dataset_args = Settings.get_settings(
                dataset_name=datasets, paper=True, minimal=True
            )
            exp_args = {
                "eval_freq": 1000,
                "get_formula": True,
                "model_path": datasets + "_" + "model.pth",
                "explain_all": True,
                "verbose": False,
                "xnum": 1000,
            }
            args = {**default_args, **exp_args, **dataset_args, **{"dataset": datasets}}
            for dedup in [True, False]:
                args["deduplicate"] = dedup
                args["experiment_id"] = experiment_id
                results = self.run(args)
                all_res.append(
                    {
                        "dataset": datasets,
                        "experiment_id": experiment_id,
                        "training_acc": results.train_acc_eval_mode[-1],
                        "test_acc": results.test_acc_eval_mode[-1],
                        # "formulas": results.formulas,
                        "deduplication": dedup,
                        "model": {
                            "num_neurons": results.args["num_neurons"],
                            "num_layers": results.args["num_layers"],
                        },
                        "encoding": {
                            "encoding_time": results.encoding_time_taken,
                            "cnf_size": results.cnf_size,
                            "eq_size": results.eq_size,
                            "deduplication": results.deduplication,
                        },
                        "xnum": results.args["xnum"],
                        "mean_explain_time": results.mean_explain_time,
                        "mean_explain_count": results.mean_explain_count,
                    }
                )
                experiment_id += 1
            with open("allres.txt", mode="w") as f:
                f.write(json.dumps(all_res, indent=4))

    def experiment(self, datasets=None, experiment_ids=None):
        all_res = []
        for dataset, experiment_id in zip(datasets, experiment_ids):
            dataset_args = Settings.get_settings(
                # dataset_name=dataset, paper=True, minimal=True
                dataset_name=dataset,
                paper=False,
                minimal=True,
            )
            exp_args = {
                "eval_freq": 1000,
                "get_formula": True,
                "model_path": dataset + "_" + "model.pth",
                "explain_all": True,
                "deduplicate": True,
                "verbose": False,
                "xnum": 30,
                # 'explain_timeout': 100,
                # TODO: add timeout
            }
            args = {**default_args, **exp_args, **dataset_args, **{"dataset": dataset}}
            args["experiment_id"] = experiment_id
            results = self.run(args)
            all_res.append(
                {
                    "dataset": dataset,
                    "experiment_id": experiment_id,
                    "training_acc": results.train_acc_eval_mode[-1],
                    "test_acc": results.test_acc_eval_mode[-1],
                    # "formulas": results.formulas,
                    "model": {
                        "num_neurons": results.args["num_neurons"],
                        "num_layers": results.args["num_layers"],
                    },
                    "encoding": {
                        "encoding_time": results.encoding_time_taken,
                        "cnf_size": results.cnf_size,
                        "eq_size": results.eq_size,
                        "deduplication": results.deduplication,
                    },
                    "xnum": results.args["xnum"],
                    "mean_explain_time": results.mean_explain_time,
                    "mean_explain_count": results.mean_explain_count,
                    "memory_usage": results.memory_usage,
                }
            )
        with open("allres.txt", mode="w") as f:
            f.write(json.dumps(all_res, indent=4))

    ####################################################################################################################

    def train_model(self, args, results, train_loader, test_loader):
        model, loss_fn, optim = get_model(args, results)

        if not args.save_model and not args.load_model:
            try:
                model.load_state_dict(torch.load(args.model_path))
            except Exception as e:
                print(e)
                train_eval(
                    args,
                    train_loader,
                    None,
                    test_loader,
                    model,
                    loss_fn,
                    optim,
                    results,
                )
                torch.save(model.state_dict(), args.model_path)
            multi_eval(
                model,
                train_loader,
                None,
                test_loader,
                results=results,
                packbits_eval=args.packbits_eval,
                extensive_eval=args.extensive_eval,
            )

        if args.load_model:
            print("Loading Model...")
            model.load_state_dict(torch.load(args.model_path))

        ####################################################################################################################
        if results is not None:
            results.store_custom("model_complete_time", time.time())

        if args.save_model:
            torch.save(model.state_dict(), args.model_path)

        if args.compile_model:
            compile_model(args, model, test_loader)

        return model

    def run(self, args):
        args = argparse.Namespace(**args)

        if args.deduplicate:
            Args["Deduplicate"] = True  # TODO: find ways to store global args
        setup_logger(args)
        seed_all(args.seed)

        logger = logging.getLogger()
        results = get_results(args.experiment_id, args)

        train_loader, test_loader, get_raw, get_train, get_test = load_dataset(args)

        def get(idx, train=True):
            if get_raw is not None:
                return get_raw(idx)
            elif train:
                return get_train(idx)
            else:
                return get_test(idx)

        model = self.train_model(args, results, train_loader, test_loader)

        Stat.start_memory_usage()
        if args.get_formula:
            dataset = get_dataset(args.dataset)
            encoding = Encoding(
                model,
                dataset,
                enc_type=Experiment.get_enc_type(args.enc_type),
            )
            if results is not None:
                results.store_encoding(encoding)

            if args.verbose:
                encoding.print()
                logger.info("\n")

            # TODO: add test to conduct this
            # Validator.validate_with_truth_table(encoding=encoding, model=model)

            # ============= ============= ============= ============= ============= ============= ============= =============

            explainer = Explainer(encoding)

            if args.explain is not None:
                raw = args.explain.split(",")
                logger.info("Raw: %s\n", raw)
                instance = Instance.from_encoding(encoding=encoding, raw=raw)
                explainer.explain_both_and_assert(instance, xnum=args.xnum)
            elif args.explain_all:
                all_times = 0
                exp_count = 0
                count = 0
                for batch, label, idx in tqdm(test_loader):
                    start = time.time()
                    for feat, i in tqdm(zip(batch, idx)):
                        raw = get(i, test=True)
                        logger.info("Raw: %s\n", raw)

                        instance = Instance.from_encoding(encoding=encoding, feat=feat)
                        exp_count_axp_plus_cxp = explainer.explain_both_and_assert(
                            instance, xnum=args.xnum
                        )
                        exp_count += exp_count_axp_plus_cxp
                    all_times += time.time() - start
                    count += len(batch)

                for batch, label, idx in tqdm(train_loader):
                    start = time.time()
                    for feat, i in tqdm(zip(batch, idx)):
                        raw = get(i, train=True)
                        logger.info("Raw: %s\n", raw)

                        instance = Instance.from_encoding(encoding=encoding, feat=feat)
                        exp_count_axp_plus_cxp = explainer.explain_both_and_assert(
                            instance, xnum=args.xnum
                        )
                        exp_count += exp_count_axp_plus_cxp
                    all_times += time.time() - start
                    count += len(batch)

                results.store_custom("mean_explain_time", all_times / count)
                results.store_custom("mean_explain_count", exp_count / count)

            elif args.explain_one:
                batch, label, idx = next(iter(test_loader))
                for feat, index in zip(batch, idx):

                    raw = get(index, test=True)
                    logger.info("Raw: %s\n", raw)

                    instance = Instance.from_encoding(encoding=encoding, feat=feat)
                    explainer.explain_both_and_assert(instance, xnum=args.xnum)

                    break
            else:
                test_count = 0
                for batch, label, idx in test_loader:
                    for feat, index in zip(batch, idx):

                        raw = get(index, test=True)
                        logger.info("Raw: %s\n", raw)
                        instance = Instance.from_encoding(encoding=encoding, feat=feat)
                        test_count += explainer.explain_both_and_assert(
                            instance, xnum=args.xnum
                        )
                print("Test Count: ", test_count)
        if results is not None:
            results.store_custom("deduplication", Stats["deduplication"])
            results.store_custom("memory_usage", Stat.get_memory_usage())
            results.save()

        return results
