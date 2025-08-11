import argparse
from attr import dataclass


@dataclass
class ModelArgs:
    connections: str = "unique"
    num_neurons: int = None
    num_layers: int = None


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
