import argparse
from attr import dataclass


@dataclass
class PySatArgs:
    enc_type_at_least: str = "tot"
    enc_type_eq: str = "lad"
    solver_type: str = "g3"
    h_type: str = "sorted"  # "lbx" or "sorted" or "sat"
    h_solver: str = "mgh"  # "mgh" or "cd195" or "g3"
    process_rounds: int = 0


def add_pysat_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--enc_type_at_least",
        type=str,
        default="mtot",
        choices=[
            "pw",
            "seqc",
            "cardn",
            "sortn",
            "tot",
            "mtot",
            "kmtot",
            "bit",
            "lad",
            "native",
        ],
        help="Encoding type for the model",
    )

    parser.add_argument(
        "--solver_type",
        type=str,
        default="g3",
        choices=["g3", "cd", "m22"],
        help="Solver type for the model",
    )

    parser.add_argument(
        "--enc_type_eq",
        type=str,
        default="pw",
        choices=[
            "pw",
            "seqc",
            "cardn",
            "sortn",
            "tot",
            "mtot",
            "kmtot",
            "bit",
            "lad",
            "native",
        ],
        help="Encoding type for equality constraints",
    )

    parser.add_argument(
        "--h_type",
        type=str,
        default="lbx",
        choices=["lbx", "sorted", "sat"],
        help="Hitting set type",
    )

    parser.add_argument(
        "--h_solver",
        type=str,
        default="mgh",
        choices=["mgh", "cd195", "g3"],
        help="Hitting set solver",
    )

    parser.add_argument(
        "--process_rounds",
        type=int,
        default=0,
        help="Number of process rounds for cnf simplification",
    )
