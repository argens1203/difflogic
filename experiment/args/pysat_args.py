import argparse
from attr import dataclass


@dataclass
class PySatArgs:
    enc_type: str = "tot"


def add_pysat_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--enc_type",
        type=str,
        default="tot",
        choices=["pw", "seqc", "cardn", "sortn", "tot", "mtot", "kmtot"],
        help="Encoding type for the model",
    )
