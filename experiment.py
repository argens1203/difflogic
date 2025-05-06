import random
import logging
import argparse
import numpy as np
import torch

from lgn.encoding import Encoding
from lgn.explanation import Explainer, Instance
from lgn.dataset import (
    input_dim_of_dataset,
    num_classes_of_dataset,
    get_attribute_ranges,
    new_load_dataset as load_dataset,
)
from lgn.model import get_model, compile_model, train_eval
from lgn.util import get_args, get_results, setup_logger
from constant import Args
from pysat.card import EncType

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
    "architecture": "randomly_connected",
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

    def debug(self, args=None):
        if args.dataset is None:
            args.dataset = "iris"
        dataset_args = Settings.debug_network_param.get(args.dataset)
        exp_args = {
            "eval_freq": 1000,
            "get_formula": True,
            "model_path": args.dataset + "_" + "model.pth",
        }
        args = {**default_args, **exp_args, **dataset_args, **args}
        self.run(args)

    def run_with_cmd(self):
        args = get_args()
        dataset_args = Settings.debug_network_param.get(args.dataset)
        exp_args = {
            "eval_freq": 1000,
            "get_formula": True,
            "model_path": args.dataset + "_" + "model.pth",
        }
        args = {**vars(args), **default_args, **exp_args, **dataset_args}
        self.run(args)

    def run(self, args):
        args = argparse.Namespace(**args)
        if args.deduplicate:
            Args["Deduplicate"] = True  # TODO: find ways to store global args
        setup_logger(args)
        seed_all(args.seed)

        logger = logging.getLogger()
        results = get_results(args.experiment_id, args)

        ####################################################################################################################

        (
            train_loader,
            test_loader,
            train_set,
            test_set,
            get_raw,
            get_train_raw,
            get_test_raw,
        ) = load_dataset(args)

        model, loss_fn, optim = get_model(args, results)

        if args.load_model:
            print("Loading Model...")
            model.load_state_dict(torch.load(args.model_path))
        else:
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

        ####################################################################################################################

        if args.save_model:
            torch.save(model.state_dict(), args.model_path)

        if args.compile_model:
            compile_model(args, model, test_loader)

        if args.get_formula:
            input_dim = input_dim_of_dataset(args.dataset)
            output_dim = num_classes_of_dataset(args.dataset)
            dataset = get_attribute_ranges(args.dataset)
            encoding = Encoding(
                model,
                input_dim,
                output_dim,
                dataset,
                enc_type=Experiment.get_enc_type(args.enc_type),
            )
            encoding.print()
            logger.info("\n")

            # TODO: add test to conduct this
            # Validator.validate_with_truth_table(encoding=encoding, model=model)

            # ============= ============= ============= ============= ============= ============= ============= =============

            explainer = Explainer(encoding)

            def get(index, train=True):
                if get_raw is not None:
                    return get_raw(index)
                if train:
                    return get_train_raw(index)
                else:
                    return get_test_raw(index)

            if args.explain is not None:
                raw = args.explain.split(",")
                logger.info("Raw: %s\n", raw)
                instance = Instance.from_encoding(encoding=encoding, raw=raw)
                explainer.explain_both_and_assert(instance, xnum=args.xnum)
            elif args.explain_all:
                for batch, label, idx in test_loader:
                    for feat, i in zip(batch, idx):
                        raw = get(i, train=False)
                        logger.info("Raw: %s\n", raw)

                        instance = Instance.from_encoding(encoding=encoding, feat=feat)
                        explainer.explain_both_and_assert(instance, xnum=args.xnum)

                for batch, label, idx in train_loader:
                    for feat, i in zip(batch, idx):
                        raw = get(i, train=True)
                        logger.info("Raw: %s\n", raw)

                        instance = Instance.from_encoding(encoding=encoding, feat=feat)
                        explainer.explain_both_and_assert(instance, xnum=args.xnum)

            elif args.explain_one:
                batch, label, idx = next(iter(test_loader))
                for feat, index in zip(batch, idx):

                    raw = get(index, train=False)
                    logger.info("Raw: %s\n", raw)

                    instance = Instance.from_encoding(encoding=encoding, feat=feat)
                    explainer.explain_both_and_assert(instance, xnum=args.xnum)

                    break
            else:
                test_count = 0
                for batch, label, idx in test_loader:
                    for feat, index in zip(batch, idx):

                        raw = get(index, train=False)
                        logger.info("Raw: %s\n", raw)
                        instance = Instance.from_encoding(encoding=encoding, feat=feat)
                        test_count += explainer.explain_both_and_assert(
                            instance, xnum=args.xnum
                        )
                print("Test Count: ", test_count)
        if results is not None:
            results.save()
            try:
                results.close()
            except Exception as e:
                print("Error closing results: ", e)
