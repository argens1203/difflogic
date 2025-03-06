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

    if args.compile_model:
        compile_model(args, model, test_loader)

    if args.get_formula:
        input_dim = input_dim_of_dataset(args.dataset)
        output_dim = num_classes_of_dataset(args.dataset)
        p_model = PseudoModel(model, input_dim, output_dim)
        p_model.check(model)

        p_model.print()
        p_model.try_it()

        # new_model, loss_fn, optim = get_model(args, results)
        # new_p_model = PseudoModel(new_model, input_dim, output_dim)
        # new_p_model.print()
        # new_p_model.check(new_model)
