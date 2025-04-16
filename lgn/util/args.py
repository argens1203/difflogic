import argparse


def get_args():
    ####################################################################################################################

    parser = argparse.ArgumentParser(
        description="Train logic gate network on the various datasets."
    )

    parser.add_argument("-eid", "--experiment_id", type=int, default=None)

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "adult",
            "breast_cancer",
            "iris",
            "monk1",
            "monk2",
            "monk3",
            "mnist",
            "mnist20x20",
            "cifar-10-3-thresholds",
            "cifar-10-31-thresholds",
            "caltech101",
        ],
        required=True,
        help="the dataset to use",
    )
    parser.add_argument(
        "--tau", "-t", type=float, default=10, help="the softmax temperature tau"
    )
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed (default: 0)")
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="batch size (default: 128)"
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.01,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--training-bit-count",
        "-c",
        type=int,
        default=32,
        help="training bit count (default: 32)",
    )

    parser.add_argument(
        "--implementation",
        type=str,
        default="cuda",
        choices=["cuda", "python"],
        help="`cuda` is the fast CUDA implementation and `python` is simpler but much slower "
        "implementation intended for helping with the understanding.",
    )

    parser.add_argument(
        "--packbits_eval",
        action="store_true",
        help="Use the PackBitsTensor implementation for an " "additional eval step.",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Compile the final model with C for CPU.",
    )

    parser.add_argument(
        "--num-iterations",
        "-ni",
        type=int,
        default=100_000,
        help="Number of iterations (default: 100_000)",
    )
    parser.add_argument(
        "--eval-freq",
        "-ef",
        type=int,
        default=2_000,
        help="Evaluation frequency (default: 2_000)",
    )

    parser.add_argument(
        "--valid-set-size",
        "-vss",
        type=float,
        default=0.0,
        help="Fraction of the train set used for validation (default: 0.)",
    )
    parser.add_argument(
        "--extensive-eval",
        action="store_true",
        help="Additional evaluation (incl. valid set eval).",
    )

    parser.add_argument(
        "--connections", type=str, default="unique", choices=["random", "unique"]
    )
    parser.add_argument("--architecture", "-a", type=str, default="randomly_connected")
    parser.add_argument("--num_neurons", "-k", type=int)
    parser.add_argument("--num_layers", "-l", type=int)

    parser.add_argument("--grad-factor", type=float, default=1.0)

    parser.add_argument(
        "--get_formula", action="store_true", help="Gets the formula of a model"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Sets vebosity"
    )

    parser.add_argument(
        "--model_path", type=str, default="model.pth", help="Path to save the model"
    )

    parser.add_argument(
        "--save_model", action="store_true", default=False, help="Save the model"
    )
    parser.add_argument(
        "--load_model", action="store_true", default=False, help="Load the model"
    )

    parser.add_argument(
        "--explain",
        type=str,
        default=None,
        help="Explain the prediction for a given input",
    )

    parser.add_argument(
        "--explain_all",
        action="store_true",
        default=False,
        help="Explain all predictions (Default: Explain only on test set)",
    )

    parser.add_argument("--explain_one", action="store_true", default=False)

    parser.add_argument("--xnum", type=int, default=1000)

    parser.add_argument(
        "--enc_type",
        type=str,
        default="tot",
        choices=["pw", "seqc", "cardn", "sortn", "tot", "mtot", "kmtot"],
        help="Encoding type for the model",
    )

    parser.add_argument("--deduplicate", action="store_true", default=False)

    args = parser.parse_args()

    if args.verbose:
        print(vars(args))

    assert (
        args.num_iterations % args.eval_freq == 0
    ), f"iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})"

    return args
