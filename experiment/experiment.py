import argparse
import torch
import json

from .args import get_args, DefaultArgs
from .helpers import Context

from lgn.encoding import Validator, encoding
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
            "verbose": "info",
            # "verbose": "debug",
            "deduplicate": "sat",  # 'bdd', 'sat', None
            "experiment_id": 10000,
            "save_model": True,
            "load_model": True,
            "max_time": 30,
            # "model_path": dataset + "_" + "model.pth",
            # "explain_inp": "3,4,7,11,13,-17,16,-15,-14,-12,-2,-9,-8,-10,-6,-5,-1",
            # "xnum": 10000,
            #  ------
            # "model_path": "model-paths/$" + dataset + "_" + "model.pth",
            # "save_model": False,
            # "num_layers": 5,
            # "num_neurons": 24,
            #  ------
            # "model_path": "model-paths/$adult" + "_" + "model.pth",
            # "save_model": False,
            # "num_layers": 5,
            # "num_neurons": 58,
            #  ------
            # "explain_one": True,
            # "explain_inp": "1,3,6,7,-2,-4,-5,-8",
            # {2, 3, 6, 8, -7, -5, -4, -1}
        }
        args = {
            **vars(default_args),
            **dataset_args,
            **exp_args,
            **{"dataset": dataset},
        }

        # Experiment.compare_encoders(args)

        results = Experiment.run(args)

        args["deduplicate"] = "bdd"
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
        if args.verbose != "warn":
            print("args:", args)
            # input("Press Enter to continue...")

        ctx = Context(args)
        # Asserts that results is not None, and enforces that entire test_set is explained
        model = Model.get_model(args, ctx=ctx)

        with ctx.use_memory_profile() as profile_memory:
            # ctx.start_memory_usage()

            encoding = Encode.get_encoding(
                model=model,
                args=args,
                ctx=ctx,
            )
            ctx.store_clause(
                encoding.get_cnf_clauses() + encoding.get_eq_constraints_clauses()
            )
            profile_memory("encoding")
            explainer = Explainer(encoding, ctx=ctx)

            if args.explain is not None:
                assert type(args.explain) == str
                raw = args.explain.split(",")
                f_exp = lambda: Explain.explain_raw(
                    args, explainer, encoding, ctx, raw=raw
                )
            elif args.explain_inp is not None:
                inp = args.explain_inp.split(",")
                inp = [int(i) for i in inp]
                inp = sorted(inp, key=lambda x: abs(x))
                f_exp = lambda: Explain.explain_raw(
                    args, explainer, encoding, ctx, inp=inp
                )
            elif args.explain_one:
                f_exp = lambda: Explain.explain_one(args, explainer, encoding, ctx)
            elif args.explain_all:
                f_exp = lambda: Explain.explain_all(args, explainer, encoding, ctx)
            else:
                f_exp = lambda: Explain.explain_dataloader(
                    ctx.test_loader,
                    args,
                    explainer=explainer,
                    encoding=encoding,
                    is_train=False,
                    ctx=ctx,
                )

            total_time_taken, exp_count, count = f_exp()
            # ============= ============= ============= ============= ============= ============= ============= =============

            ctx.results.store_explanation_stat(exp_count / count, ctx.deduplication)
            ctx.results.store_resource_usage(total_time_taken / exp_count, -1)
            profile_memory("explanation")
            ctx.results.store_explanation_ready_time()
            ctx.results.store_counts(count, exp_count)
        # ctx.end_memory_usage()
        ctx.results.save()
        ctx.results.store_end_time()
        ctx.display()
        # input("Press Enter to continue...")

        return ctx.results

    @staticmethod
    def compare_encoders(args):
        args = DefaultArgs(**args)

        ctx = Context(args)
        model = Model.get_model(args, ctx=ctx)
        ctx.debug(lambda: [layer.print() for layer in model])

        args.deduplicate = None
        encoding1 = Encode.get_encoding(
            model=model,
            args=args,
            ctx=ctx,
        )
        ctx.debug(encoding1.print)

        args.deduplicate = "bdd"
        encoding2 = Encode.get_encoding(
            model=model,
            args=args,
            ctx=ctx,
        )
        ctx.debug(encoding2.print)

        args.deduplicate = "sat"
        encoding3 = Encode.get_encoding(
            model=model,
            args=args,
            ctx=ctx,
        )
        ctx.debug(encoding3.print)

        assert str(encoding2.formula) == str(encoding3.formula), (
            "Formulas should be equal",
            encoding2.formula,
            encoding3.formula,
        )
        validator = Validator(ctx)

        validator.validate_encodings_with_data(
            encoding1=encoding1, encoding2=encoding2, dataloader=ctx.test_loader
        )
        validator.validate_encodings_with_data(
            encoding1=encoding1, encoding2=encoding3, dataloader=ctx.test_loader
        )
        validator.validate_encodings_with_data(
            encoding1=encoding2, encoding2=encoding3, dataloader=ctx.test_loader
        )

        validator.validate_encodings_with_truth_table(
            encoding1=encoding1, encoding2=encoding2, dataset=ctx.dataset
        )
        validator.validate_encodings_with_truth_table(
            encoding1=encoding1, encoding2=encoding3, dataset=ctx.dataset
        )
        validator.validate_encodings_with_truth_table(
            encoding1=encoding2, encoding2=encoding3, dataset=ctx.dataset
        )

        explainer = Explainer(encoding1, ctx=ctx)
        f_exp = lambda: Explain.explain_all(args, explainer, encoding1, ctx)
        f_exp()

        explainer = Explainer(encoding2, ctx=ctx)
        f_exp = lambda: Explain.explain_all(args, explainer, encoding2, ctx)
        f_exp()

        explainer = Explainer(encoding3, ctx=ctx)
        f_exp = lambda: Explain.explain_all(args, explainer, encoding3, ctx)
        f_exp()
