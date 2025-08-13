import argparse
import torch
import json

from .args import get_args, DefaultArgs
from .helpers import Context

from lgn.encoding import Validator
from lgn.explanation import Explainer

from .explain import Explain
from .settings import Settings
from .encode import Encode
from .model import Model

default_args = DefaultArgs()

torch.set_num_threads(1)  # ???


class Experiment:
    @staticmethod
    def debug(dataset=None):
        dataset = dataset if dataset is not None else "iris"
        dataset_args: dict[str, int] = Settings.debug_network_param.get(dataset) or {}
        exp_args = {
            "eval_freq": 1000,
            "model_path": dataset + "_" + "model.pth",
            "verbose": False,
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

    ####################################################################################################################

    @staticmethod
    def find_model():
        experiment_id = 1000
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
                        "experiment_id": experiment_id,
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

                    args["experiment_id"] = experiment_id

                    args = argparse.Namespace(**args)
                    ctx = Context(args)
                    model = Model.get_model(args, ctx=ctx)
                    ctx.results.save()
                    results = ctx.results
                    if results.test_acc > best_acc:
                        best_acc = results.test_acc
                        best_eid = experiment_id
                        torch.save(model.state_dict(), args.model_path)

                    experiment_id += 1
            best_ids.append((dataset, best_eid, best_acc))
        with open("best_ids.txt", mode="w") as f:
            f.write(json.dumps(best_ids, indent=4))
        print(best_ids)

    @staticmethod
    def rerun_experiments(experiment_ids=[], output_eid=500):
        # experiement_ids = [1000, 1015, 1031, 1048, 1060]
        for eid in experiment_ids:
            for deduplication in [False, True]:
                with open(f"results/0000{eid}.json", mode="r") as f:
                    results = json.load(f)
                    args = results["args"]
                    args["load_model"] = True
                    args["experiment_id"] = output_eid
                    args["deduplicate"] = deduplication

                    Experiment.run(args)

                    output_eid += 1

    @staticmethod
    def run(args):
        args = DefaultArgs(**args)
        # args = argparse.Namespace(**args)
        print("args:", args)
        # input("Press Enter to continue...")

        ctx = Context(args)
        # Asserts that results is not None, and enforces that entire test_set is explained
        model = Model.get_model(args, ctx=ctx)

        ctx.start_memory_usage()

        encoding = Encode.get_encoding(
            model=model,
            args=args,
            ctx=ctx,
        )

        explainer = Explainer(encoding, ctx=ctx)

        total_time_taken, exp_count, count = Explain.explain_dataloader(
            ctx.test_loader,
            args,
            explainer=explainer,
            encoding=encoding,
            is_train=False,
            ctx=ctx,
        )
        # ============= ============= ============= ============= ============= ============= ============= =============

        ctx.results.store_explanation_stat(exp_count / count, ctx.deduplication)
        ctx.results.store_resource_usage(
            total_time_taken / exp_count, ctx.get_memory_usage()
        )
        ctx.results.store_counts(count, exp_count)
        ctx.end_memory_usage()
        ctx.results.save()

        return ctx.results

    @staticmethod
    def compare_encoders(args):
        args = DefaultArgs(**args)

        ctx = Context(args)
        model = Model.get_model(args, ctx=ctx)
        for layer in model:
            layer.print()

        args.deduplicate = "bdd"
        encoding2 = Encode.get_encoding(
            model=model,
            args=args,
            ctx=ctx,
        )
        encoding2.print()
        input("Press enter to continue...")

        args.deduplicate = "sat"
        encoding3 = Encode.get_encoding(
            model=model,
            args=args,
            ctx=ctx,
        )
        encoding3.print()
        input("Press enter to continue...")

        args.deduplicate = None
        encoding1 = Encode.get_encoding(
            model=model,
            args=args,
            ctx=ctx,
        )
        encoding1.print()

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
