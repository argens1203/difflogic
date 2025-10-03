import argparse
from attr import dataclass
from typing import Optional


@dataclass
class EncodingArgs:
    deduplicate: Optional[str] = None
    strategy: str = "full"
    ohe_deduplication: bool = True


def add_encoding_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--deduplicate",
        type=str,
        default=None,
        choices=["sat", "bdd"],
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="full",
        choices=["full", "b_full", "parent"],
        help="Deduplication strategy to use (only for SAT).",
    )

    parser.add_argument(
        "--ohe_deduplication",
        action="store_true",
        default=True,
        help="Enable OHE deduplication (only for SAT).",
    )
