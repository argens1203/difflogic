import argparse
from attr import dataclass


@dataclass
class PySatArgs:
    enc_type: str = "tot"
    solver_type: str = "g3"


def add_pysat_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--enc_type",
        type=str,
        default="tot",
        choices=["pw", "seqc", "cardn", "sortn", "tot", "mtot", "kmtot"],
        help="Encoding type for the model",
    )

    parser.add_argument(
        "--solver_type",
        type=str,
        default="g3",
        choices=["g3", "cd", "m22"],
        help="Solver type for the model",
    )
