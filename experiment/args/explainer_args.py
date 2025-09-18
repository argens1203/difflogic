import argparse
from attr import dataclass
from typing import Optional


@dataclass
class ExplainerArgs:
    xnum: int = 1000
    max_explain_time: int = 3600
    explain: Optional[str] = None
    explain_all: bool = False
    explain_one: bool = False
    explain_inp: Optional[str] = None


def add_explainer_args(parser: argparse.ArgumentParser):
    parser.add_argument("--xnum", type=int, default=1000)
    parser.add_argument(
        "--max_explain_time",
        type=int,
        default=3600,
        help="Timeout for entire explanation (in seconds) (default: 3600)",
    )
    parser.add_argument(
        "--explain",
        type=str,
        default=None,
        help="Explain the prediction for a given input (raw)",
    )
    parser.add_argument(
        "--explain_inp",
        type=str,
        default=None,
        help="Explain the prediction of a given (boolean) input (eg.: 1,-2,3,-4)",
    )
    parser.add_argument(
        "--explain_all",
        action="store_true",
        default=False,
        help="Explain all predictions (Default: Explain only on test set)",
    )
    parser.add_argument("--explain_one", action="store_true", default=False)
