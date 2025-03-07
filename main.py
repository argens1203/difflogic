import random

import numpy as np
import torch

from lgn.pseudo_model import PseudoModel
from lgn.dataset import (
    load_dataset,
    input_dim_of_dataset,
    num_classes_of_dataset,
)
from lgn.model import get_model, compile_model
from lgn.trainer import train_eval
from lgn.args import get_args
from lgn.util import get_results

torch.set_num_threads(1)  # ???


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = get_args()

    results = get_results(args.experiment_id, args)
    seed_all(args.seed)

    ####################################################################################################################

    train_loader, validation_loader, test_loader = load_dataset(args)
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
        p_model = PseudoModel(model, input_dim, output_dim)
        p_model.check(model)

        # p_model.print(print_vpool=True)
        p_model.pairwise_comparisons(1, 2, inp=[-1, -2, -3, -4, 5, -6, -7, -8])
        p_model.pairwise_comparisons(1, 2, inp=[-1, -2, -3, -4, 5, -6, -7, -8])
        p_model.pairwise_comparisons(1, 3, inp=[-1, -2, -3, -4, 5, -6, -7, -8])
        # p_model.print(print_vpool=True)

        # ============= ============= ============= ============= ============= ============= ============= =============

        instance = train_loader.dataset[0]
        feat, label = instance
        print(instance)
        print(feat.to(int))

        def feat_to_input(feat):
            inp = [idx + 1 if f == 1 else -(idx + 1) for idx, f in enumerate(feat)]
            return inp

        inp = feat_to_input(feat)
        print(inp)

        def explain(feat):
            true_class = p_model.predict_votes(feat)
            print(true_class)

        explain([inp])

        #     for cls in range(1, 4):
        #         for

        # ============= ============= ============= ============= ============= ============= ============= =============

        # new_model, loss_fn, optim = get_model(args, results)
        # new_p_model = PseudoModel(new_model, input_dim, output_dim)
        # new_p_model.print()
        # new_p_model.check(new_model)
