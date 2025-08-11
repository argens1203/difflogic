import argparse
from attr import dataclass


@dataclass
class ExplainerArgs:
    xnum: int = 1000
    max_time: int = 3600
    explain: str = None
    explain_all: bool = False
    explain_one: bool = False


def add_explainer_args(parser: argparse.ArgumentParser):
    parser.add_argument("--xnum", type=int, default=1000)
    parser.add_argument(
        "--max_time",
        type=int,
        default=3600,
        help="Timeout for entire run (in seconds) (default: 3600)",
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
