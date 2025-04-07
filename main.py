import random
import logging

import numpy as np
import torch

from lgn.encoding import Encoding
from lgn.explanation import Explainer, Instance
from lgn.dataset import Binarizer
from lgn.dataset.dataset import (
    load_dataset,
    IrisDataset,
    input_dim_of_dataset,
    num_classes_of_dataset,
    get_attribute_ranges,
    Caltech101Dataset,
    MNISTDataset,
)
from lgn.model import get_model, compile_model
from lgn.model import train_eval
from lgn.util import get_args
from lgn.util import get_results, setup_logger

torch.set_num_threads(1)  # ???


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def new_load_dataset(args):
    if args.dataset == "iris":
        dataset = IrisDataset(transform=Binarizer(IrisDataset(), 2))
    elif args.dataset == "caltech101":
        dataset = Caltech101Dataset()
    elif args.dataset == "adult":
        return load_dataset(args)
    elif args.dataset in ["monk1", "monk2", "monk3"]:
        return load_dataset(args)
    elif args.dataset == "breast_cancer":
        return load_dataset(args)
    elif args.dataset == "mnist":
        dataset = MNISTDataset()

    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=int(1e6), shuffle=False
    )
    validation_loader = None
    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    args = get_args()
    args.model_path = args.dataset + "_" + args.model_path
    args.batch_size = 100
    args.num_iterations = 2000
    args.eval_freq = 1000
    args.num_layers = 2
    args.get_formula = True

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

    setup_logger(args)
    logger = logging.getLogger()

    # logging.disable(logging.DEBUG)

    results = get_results(args.experiment_id, args)
    # seed_all(args.seed)
    seed_all(0)

    ####################################################################################################################

    # dataset = AdultDataset(transform=Binarizer(AdultDataset(), 2))
    train_loader, validation_loader, test_loader = new_load_dataset(args)

    # train_loader, validation_loader, test_loader = load_dataset(args)

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

        # instance = train_loader.dataset[0]
        # feat, label = instance

        # Explainer(encoding).explain(feat)

        # NEW
        print(encoding.input_ids)
        print(encoding.input_handles)
        print(encoding.get_attribute_ranges())
        # NEW

        explainer = Explainer(encoding)

        from operator import itemgetter

        def explain_both_and_assert(instance):
            explainer.explain(instance)

            axps, axp_dual = explainer.mhs_mus_enumeration(instance)
            cxps, cxp_dual = explainer.mhs_mcs_enumeration(instance)

            logger.info("Input: %s", instance.get_input())
            logger.info(
                "AXPs: %s",
                str(
                    [sorted(one) for one in sorted(axps, key=lambda x: (len(x), x[0]))]
                ),
            )
            logger.info(
                "Duals: %s",
                str(
                    [
                        sorted(one)
                        for one in sorted(axp_dual, key=lambda x: (len(x), x[0]))
                    ]
                ),
            )
            logger.info(
                "CXPs: %s",
                str(
                    [sorted(one) for one in sorted(cxps, key=lambda x: (len(x), x[0]))]
                ),
            )
            logger.info(
                "Duals: %s",
                str(
                    [
                        sorted(one)
                        for one in sorted(cxp_dual, key=lambda x: (len(x), x[0]))
                    ]
                ),
            )
            axp_set = set()
            for axp in axps:
                axp_set.add(frozenset(axp))
            cxp_set = set()
            for cxp in cxps:
                cxp_set.add(frozenset(cxp))
            axp_dual_set = set()
            for axp_d in axp_dual:
                axp_dual_set.add(frozenset(axp_d))
            cxp_dual_set = set()
            for cxp_d in cxp_dual:
                cxp_dual_set.add(frozenset(cxp_d))

            # TODO: does not work on adult (test set)
            assert axp_set.difference(cxp_dual_set) == set()
            assert cxp_dual_set.difference(axp_set) == set()

            assert axp_dual_set.difference(cxp_set) == set()
            assert cxp_set.difference(axp_dual_set) == set()

            return len(axps) + len(axp_dual)

        if args.explain is not None:
            inp = args.explain.split(",")
            inp = [int(i) for i in inp]
            print(inp)
            instance = Instance.from_encoding(encoding=encoding, inp=inp)
            # try:
            explain_both_and_assert(instance)
        elif args.explain_all:
            for batch, label in test_loader:
                for feat in batch:
                    instance = Instance.from_encoding(encoding=encoding, feat=feat)
                    explain_both_and_assert(instance)
            for batch, label in train_loader:
                for feat in batch:
                    instance = Instance.from_encoding(encoding=encoding, feat=feat)
                    explain_both_and_assert(instance)
        elif args.explain_one:
            batch, label = next(iter(test_loader))
            for feat in batch:
                instance = Instance.from_encoding(encoding=encoding, feat=feat)
                explain_both_and_assert(instance)
                break
        else:
            test_count = 0
            for batch, label in test_loader:
                for feat in batch:
                    instance = Instance.from_encoding(encoding=encoding, feat=feat)
                    test_count += explain_both_and_assert(instance)
            print("Test Count: ", test_count)


# First Run "python main.py  -bs 100 --dataset iris -ni 2000 -ef 1_000 -k 6 -l 2 --get_formula --save_model"
# Subsequent Run "python main.py  -bs 100 --dataset iris -ni 2000 -ef 1_000 -k 6 -l 2 --get_formula --load_model"
