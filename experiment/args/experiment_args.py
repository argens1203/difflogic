import argparse
from attr import dataclass
from typing import Optional


@dataclass
class ExperimentArgs:
    experiment_id: Optional[int] = None
    dataset: str = "iris"
    verbose: str = "info"
    save_model: bool = True
    load_model: bool = True
    model_path: str = "model.pth"


def add_experiment_args(parser: argparse.ArgumentParser):
    parser.add_argument("-eid", "--experiment_id", type=int, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "adult",
            "breast_cancer",
            "iris",
            "monk1",
            "monk2",
            "monk3",
            "mnist",
            "mnist20x20",
            "cifar-10-3-thresholds",
            "cifar-10-31-thresholds",
            "caltech101",
        ],
        required=True,
        help="the dataset to use",
    )
    parser.add_argument(
        "--verbose",
        choices=["debug", "info", "warn"],
        type=str,
        default="info",
        help="Sets verbosity",
    )
    parser.add_argument(
        "--save_model", action="store_true", default=True, help="Save the model"
    )
    parser.add_argument(
        "--load_model", action="store_true", default=True, help="Load the model"
    )
    parser.add_argument(
        "--model_path", type=str, default="model.pth", help="Path to save the model"
    )
