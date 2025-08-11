import argparse
from attr import dataclass


@dataclass
class SettingsArgs:
    seed: int = 0
    packbits_eval: bool = False
    compile_model: bool = False
    implementation: str = "cuda"


def add_settings_args(parser: argparse.ArgumentParser):
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed (default: 0)")
    parser.add_argument(
        "--packbits_eval",
        action="store_true",
        help="Use the PackBitsTensor implementation for an " "additional eval step.",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Compile the final model with C for CPU.",
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="cuda",
        choices=["cuda", "python"],
        help="`cuda` is the fast CUDA implementation and `python` is simpler but much slower "
        "implementation intended for helping with the understanding.",
    )
