import argparse
from attr import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    connections: str = "unique"
    num_neurons: Optional[int] = None
    num_layers: Optional[int] = None


def add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--connections",
        "-c",
        type=str,
        default="unique",
        choices=[
            "unique",
            #  "random"
        ],
    )
    parser.add_argument("--num_neurons", "-k", type=int)
    parser.add_argument("--num_layers", "-l", type=int)
