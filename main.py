import random
import logging
import os

import numpy as np
import torch

from lgn.encoding import Validator, Encoding
from lgn.explanation import Explainer
from lgn.dataset.dataset import (
    load_dataset,
    CustomDataset,
    Binarizer,
    input_dim_of_dataset,
    num_classes_of_dataset,
)
from lgn.model import get_model, compile_model
from lgn.model import train_eval
from lgn.util import get_args
from lgn.util import get_results

torch.set_num_threads(1)  # ???

console_handler = logging.StreamHandler()
console_format = logging.Formatter(
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    "%(message)s"
)
console_handler.setFormatter(console_format)

LOG_FILE_PATH = "main.log"


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = get_args()
    args.batch_size = 100
    args.dataset = "iris"
    args.num_iterations = 2000
    args.eval_freq = 1000
    args.num_neurons = 6
    args.num_layers = 2
    args.get_formula = True

    if args.verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
    else:
        if os.path.exists(LOG_FILE_PATH):
            os.remove(LOG_FILE_PATH)
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_format = logging.Formatter(
            # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            "%(message)s"
        )
        file_handler.setFormatter(file_format)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    results = get_results(args.experiment_id, args)
    # seed_all(args.seed)
    seed_all(0)

    ####################################################################################################################

    # dataset = CustomDataset(transform=Binarizer(CustomDataset(), 2))
    # train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=args.batch_size, shuffle=True
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_set, batch_size=int(1e6), shuffle=False
    # )
    # validation_loader = None

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
        encoding = Encoding(model, input_dim, output_dim)
        encoding.print()

        Validator.validate_with_truth_table(encoding=encoding, model=model)

        # ============= ============= ============= ============= ============= ============= ============= =============

        # instance = train_loader.dataset[0]
        # feat, label = instance

        # Explainer(encoding).explain(feat)

        explainer = Explainer(encoding)

        if args.explain is not None:
            inp = args.explain.split(",")
            inp = [int(i) for i in inp]
            try:
                explainer.explain(inp=inp)
            except Exception as e:
                logger.error(e)
        else:
            for batch, label in train_loader:
                for feat in batch:
                    # encoding.print()
                    explainer.explain(feat=feat)

# First Run "python main.py  -bs 100 --dataset iris -ni 2000 -ef 1_000 -k 6 -l 2 --get_formula --save_model"
# Subsequent Run "python main.py  -bs 100 --dataset iris -ni 2000 -ef 1_000 -k 6 -l 2 --get_formula --load_model"
