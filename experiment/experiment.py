import random
import logging
import argparse
import numpy as np
import torch
import json
from .util import get_args, setup_logger

import random
import logging
import numpy as np
import torch
from experiment.encode import Encode
from experiment.model import Model
from lgn.encoding import Validator
from lgn.explanation import Explainer
from .util import Stat

from constant import Stats
from .explain import Explain
from .context import Context

from .util import get_enc_type


torch.set_num_threads(1)  # ???


def seed_all(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


from .util import DefaultArgs
from .settings import Settings

default_args = DefaultArgs()


class Experiment:
    @staticmethod
    def debug(dataset=None):
        dataset = dataset if dataset is not None else "iris"
        dataset_args: dict[str, int] = Settings.debug_network_param.get(dataset) or {}
        exp_args = {
            "eval_freq": 1000,
            "model_path": dataset + "_" + "model.pth",
            "verbose": True,
            "save_model": True,
            "load_model": True,
            "deduplicate": None,  # 'bdd', 'sat', None
            "experiment_id": 10000,
        }
        args = {
            **vars(default_args),
            **exp_args,
            **dataset_args,
            **{"dataset": dataset},
        }

        print("args:", args)
        input("Press Enter to continue...")

        Experiment.compare_encoders(args)

        results = Experiment.run(args)

        return results

    @staticmethod
    def run_with_cmd():
        args = get_args()
        dataset_args = Settings.debug_network_param.get(args.dataset) or {}
        exp_args = {
            "model_path": args.dataset + "_" + "model.pth",
        }
        args = {**vars(default_args), **exp_args, **dataset_args, **vars(args)}

        Experiment.run(args)

    @staticmethod
    def f():
        experiment_id = 1
        all_res = []

        for datasets in ["monk1", "monk2", "monk3"]:
            dataset_args = (
                Settings.get_settings(dataset_name=datasets, paper=True, minimal=True)
                or {}
            )
            exp_args = {
                "eval_freq": 1000,
                "model_path": datasets + "_" + "model.pth",
                "explain_all": True,
                "verbose": False,
                "xnum": 1000,
            }
            args = {
                **vars(default_args),
                **exp_args,
                **dataset_args,
                **{"dataset": datasets},
            }
            for dedup in ["bdd", "sat", None]:
                args["deduplicate"] = dedup
                args["experiment_id"] = experiment_id

                results = Experiment.run(args)
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

    @staticmethod
    def experiment(datasets=[], base_experiment_id=10000):
        all_res = []
        experiment_id = base_experiment_id
        for dataset in datasets:
            for deduplicate in ["sat", "bdd"]:
                experiment_id += 1
                dataset_args = Settings.get_settings(
                    # dataset_name=dataset, paper=True, minimal=True
                    dataset_name=dataset,
                    paper=False,
                    minimal=True,
                )
                exp_args = {
                    "eval_freq": 1000,
                    "model_path": dataset + "_" + "model.pth",
                    "explain_all": False,
                    "explain_one": True,
                    "deduplicate": deduplicate,
                    "verbose": False,
                    "xnum": 30,
                    "load_model": True,
                    "save_model": True,
                    "experiment_id": experiment_id,
                    # 'explain_timeout': 100,
                    # TODO: add timeout
                }
                args = {
                    **vars(default_args),
                    **exp_args,
                    **dataset_args,
                    **{"dataset": dataset},
                }
                args["experiment_id"] = experiment_id

                results = Experiment.run(args)
                all_res.append(
                    {
                        "dataset": dataset,
                        "experiment_id": experiment_id,
                        # "training_acc": (
                        #     results.train_acc_eval_mode[-1]
                        #     if results.train_acc_eval_mode
                        #     else None
                        # ),
                        # "test_acc": (
                        #     results.test_acc_eval_mode[-1]
                        #     if results.test_acc_eval_mode
                        #     else None
                        # ),
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

    @staticmethod
    def find_model():
        experiement_id = 1000
        best_ids = []
        for dataset in [
            "iris",
            "monk1",
            "monk2",
            "monk3",
            "breast_cancer",
            "mnist",
            "adult",
        ]:
            best_acc = 0
            best_eid = 0
            for temp in [1, 1 / 0.3, 1 / 0.1, 1 / 0.03, 1 / 0.01]:
                for grad_factor in [1, 1.5, 2]:
                    dataset_args = Settings.get_settings(
                        dataset_name=dataset, paper=True, minimal=True
                    )
                    exp_args = {
                        "model_path": f"model-paths/{dataset}_model.pth",
                        "batch_size": 64,
                        "experiment_id": experiement_id,
                        "tau": temp,
                        "grad_factor": grad_factor,
                        "learning_rate": 0.01,
                        "num_iterations": 5000,
                        "eval_freq": 1000,
                        "save_model": False,
                        "load_model": False,
                    }
                    args = {
                        **vars(default_args),
                        **dataset_args,
                        **exp_args,
                        **{"dataset": dataset},
                    }
                    args = argparse.Namespace(**args)

                    setup_logger(args)
                    seed_all(args.seed)
                    args.experiment_id = experiement_id

                    ctx = Context(args)
                    model = Model.get_model(args, ctx=ctx)
                    ctx.results.save()
                    results = ctx.results
                    if results.test_acc > best_acc:
                        best_acc = results.test_acc
                        best_eid = experiement_id
                        torch.save(model.state_dict(), args.model_path)

                    experiement_id += 1
            best_ids.append((dataset, best_eid, best_acc))
        with open("best_ids.txt", mode="w") as f:
            f.write(json.dumps(best_ids, indent=4))
        print(best_ids)

    @staticmethod
    def get_and_retest_model():
        experiement_ids = [1000, 1015, 1031, 1048, 1060]
        output_eid = 500
        for eid in experiement_ids:
            for deduplication in [False, True]:
                with open(f"results/0000{eid}.json", mode="r") as f:
                    results = json.load(f)
                    args = results["args"]
                    args["load_model"] = True
                    args["save_model"] = False
                    args["experiment_id"] = output_eid
                    args["deduplicate"] = deduplication
                    args["verbose"] = False
                    args["xnum"] = 100
                    args["max_time"] = 3600

                    Experiment.run(args)

                    output_eid += 1

    @staticmethod
    def run(args):
        args = argparse.Namespace(**args)
        print("args:", args)
        input("Press Enter to continue...")

        setup_logger(args)
        seed_all(args.seed)

        ctx = Context(args)
        # Asserts that results is not None, and enforces that entire test_set is explained
        model = Model.get_model(args, ctx=ctx)

        Stat.start_memory_usage()

        encoding = Encode.get_encoding(
            model=model,
            enc_type=get_enc_type(args.enc_type),
            deduplication=args.deduplicate,
            ctx=ctx,
        )

        # Validator.validate_with_truth_table(encoding=self.encoding, model=self.model)
        encoding.print()
        explainer = Explainer(encoding)

        total_time_taken, exp_count, count = Explain.explain_dataloader(
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
    def compare_encoders(args):
        args = argparse.Namespace(**args)
        ctx = Context(args)
        model = Model.get_model(args, ctx=ctx)

        encoding2 = Encode.get_encoding(
            model=model,
            enc_type=get_enc_type(args.enc_type),
            deduplication="bdd",
            ctx=ctx,
        )
        encoding3 = Encode.get_encoding(
            model=model,
            enc_type=get_enc_type(args.enc_type),
            deduplication="sat",
            ctx=ctx,
        )
        encoding1 = Encode.get_encoding(
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
