import random
import logging
import argparse
import numpy as np
import torch
import json
from lgn.util import get_args, setup_logger

torch.set_num_threads(1)  # ???


def seed_all(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


from lgn.util import DefaultArgs
from .settings import Settings

default_args = DefaultArgs()
from .one_experiment import OneExperiment


class Experiment:
    # def __init__(self):
    # self.logger = logging.getLogger()

    def __init__(self):
        pass

    def debug(self, dataset=None):
        dataset = dataset if dataset is not None else "iris"
        dataset_args: dict[str, int] = Settings.debug_network_param.get(dataset) or {}
        print(dataset_args)
        print(type(dataset_args))
        exp_args = {
            "eval_freq": 1000,
            "model_path": dataset + "_" + "model.pth",
            "verbose": True,
        }
        args = {
            **vars(default_args),
            **exp_args,
            **dataset_args,
            **{"dataset": dataset},
        }
        self.run(args)

    # def pseudo_run(self, dataset, model_path):
    #     num_neurons = Settings.dataset_neurons.get(dataset)
    #     num_layers = 2
    #     num_iterations = 2000
    #     batch_size = 100
    #     model_path = dataset + "_" + model_path
    #     eval_freq = 1000

    def run_with_cmd(self):
        args = get_args()
        dataset_args = Settings.debug_network_param.get(args.dataset) or {}
        exp_args = {
            "eval_freq": 1000,
            "model_path": args.dataset + "_" + "model.pth",
            "verbose": True,
        }
        args = {**vars(default_args), **exp_args, **dataset_args, **vars(args)}
        self.run(args)

    def f(self):
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
                "model_path": dataset + "_" + "model.pth",
                "explain_all": True,
                "deduplicate": True,
                "verbose": False,
                "xnum": 30,
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

    def run(self, args):
        args = argparse.Namespace(**args)
        setup_logger(args)
        seed_all(args.seed)
        # one = OneExperiment(args)
        # return one.run_presentation
        args.experiment_id = 0
        one = OneExperiment(args)
        results = one.run_experiment(args)
        return results

    def find_model(self):
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
                    results, model = OneExperiment(args).find_model(args)
                    if results.test_acc > best_acc:
                        best_acc = results.test_acc
                        best_eid = experiement_id
                        torch.save(model.state_dict(), args.model_path)

                    experiement_id += 1
            best_ids.append((dataset, best_eid, best_acc))
        with open("best_ids.txt", mode="w") as f:
            f.write(json.dumps(best_ids, indent=4))
        print(best_ids)

    def get_and_retest_model(self):
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

                    args = argparse.Namespace(**args)
                    setup_logger(args)
                    seed_all(args.seed)
                    res = OneExperiment(args).run_experiment(args)

                    output_eid += 1
