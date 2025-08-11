import argparse
from attr import dataclass


@dataclass
class TrainingArgs:
    tau: int = 10
    learning_rate: float = 0.01
    grad_factor: float = 1.0
    batch_size: int = 128
    training_bit_count: int = 32  # Torch floating point precision
    eval_freq: int = 2000
    num_iterations: int = 100_000
    valid_set_size: float = 0.0
    extensive_eval: bool = False


def add_training_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tau", "-t", type=float, default=10, help="the softmax temperature tau"
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.01,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument("--grad-factor", type=float, default=1.0)
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="batch size (default: 128)"
    )
    parser.add_argument(
        "--training-bit-count",
        type=int,
        default=32,
        help="training bit count (default: 32)",
    )
    parser.add_argument(
        "--eval-freq",
        "-ef",
        type=int,
        default=2_000,
        help="Evaluation frequency (default: 2_000)",
    )
    parser.add_argument(
        "--num-iterations",
        "-ni",
        type=int,
        default=100_000,
        help="Number of iterations (default: 100_000)",
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
