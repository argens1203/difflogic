import random
import logging

import numpy as np
import torch

from lgn.encoding import Encoding
from lgn.explanation import Explainer, Instance
from lgn.dataset import (
    input_dim_of_dataset,
    num_classes_of_dataset,
    get_attribute_ranges,
    new_load_dataset,
)
from lgn.model import get_model, compile_model, train_eval
from lgn.util import get_args, get_results, setup_logger

torch.set_num_threads(1)  # ???


def seed_all(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_neurons(args):
    if args.dataset == "iris":
        args.num_neurons = 6  # >= 4, div 3
    elif args.dataset == "caltech101":
        args.num_neurons = 41 * 101  # >= 4096, div 101
    elif args.dataset == "adult":
        args.num_neurons = 58  # >= 58, div 2
    elif args.dataset in ["monk1", "monk2", "monk3"]:
        args.num_neurons = 10  # >= 9, div 2
    elif args.dataset == "breast_cancer":
        args.num_neurons = 26  # >= 26, div 2
    elif args.dataset == "mnist":
        args.num_neurons = 400  # >= 400, div 10


if __name__ == "__main__":
    args = get_args()
    args.model_path = args.dataset + "_" + args.model_path
    args.batch_size = 100
    args.num_iterations = 2000
    args.eval_freq = 1000
    args.num_layers = 2
    args.get_formula = True

    setup_neurons(args)
    setup_logger(args)
    seed_all(args.seed)

    logger = logging.getLogger()
    results = get_results(args.experiment_id, args)

    ####################################################################################################################

    train_loader, validation_loader, test_loader = new_load_dataset(args)

    model, loss_fn, optim = get_model(args, results)

    if args.load_model:
        print("Loading Model...")
        model.load_state_dict(torch.load(args.model_path))
    else:
        train_eval(
            args,
            train_loader,
            validation_loader,
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
        attribute_ranges = get_attribute_ranges(args.dataset)
        encoding = Encoding(model, input_dim, output_dim, attribute_ranges)
        encoding.print()

        # TODO: add test to conduct this
        # Validator.validate_with_truth_table(encoding=encoding, model=model)

        # ============= ============= ============= ============= ============= ============= ============= =============

        explainer = Explainer(encoding)

        if args.explain is not None:
            inp = args.explain.split(",")
            inp = [int(i) for i in inp]
            instance = Instance.from_encoding(encoding=encoding, inp=inp)
            explainer.explain_both_and_assert(instance)
        elif args.explain_all:
            for batch, label in test_loader:
                for feat in batch:
                    instance = Instance.from_encoding(encoding=encoding, feat=feat)
                    explainer.explain_both_and_assert(instance)
            for batch, label in train_loader:
                for feat in batch:
                    instance = Instance.from_encoding(encoding=encoding, feat=feat)
                    explainer.explain_both_and_assert(instance)
        elif args.explain_one:
            batch, label = next(iter(test_loader))
            for feat in batch:
                instance = Instance.from_encoding(encoding=encoding, feat=feat)
                explainer.explain_both_and_assert(instance)
                break
        else:
            test_count = 0
            for batch, label in test_loader:
                for feat in batch:
                    instance = Instance.from_encoding(encoding=encoding, feat=feat)
                    test_count += explainer.explain_both_and_assert(instance)
            print("Test Count: ", test_count)
