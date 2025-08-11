import argparse
from attr import dataclass
from typing import Optional


@dataclass
class EncodingArgs:
    deduplicate: Optional[str] = None


def add_encoding_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--deduplicate",
        type=str,
        default=None,
        choices=["sat", "bdd"],
    )
