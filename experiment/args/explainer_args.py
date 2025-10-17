import argparse
from attr import dataclass
from typing import Optional


@dataclass
class ExplainerArgs:
    xnum: Optional[int] = None
    max_explain_time: int = 3600
    explain: Optional[str] = None
    explain_all: bool = False
    explain_one: bool = False
    explain_inp: Optional[str] = None
    explain_algorithm: Optional[str] = "both"

    explain_switch_window: int = 20
    explain_switch_alpha: float = 2.0
    # explain_switch_epsilon: Optional[float] = 0.05


def add_explainer_args(parser: argparse.ArgumentParser):
    parser.add_argument("--xnum", type=int, default=None)
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
        help="Explain all predictions (Test + Train set). Warning: This can take a long time",
    )
    parser.add_argument(
        "--explain_one",
        action="store_true",
        default=False,
        help="Explain one prediction from test set. Note.: This uses fast method for getting one explanation only",
    )
    parser.add_argument(
        "--explain_algorithm",
        type=str,
        default="both",
        choices=["mus", "mcs", "var", "both", "find_one"],
        help="Explanation algorithm to use (default: both)",
    )

    parser.add_argument(
        "--explain_switch_window",
        type=float,
        default=None,
        help="The number of previous explanations to consider when switching explanation algorithms (default: 20)",
    )

    parser.add_argument(
        "--explain_switch_alpha",
        type=float,
        default=None,
        help="The ratio of avg size of CXPs over AXPs for switching explanation algorithms (default: 2.0)",
    )
