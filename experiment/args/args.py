import argparse
from attr import dataclass

from .encoding_args import *
from .experiment_args import *
from .explainer_args import *
from .model_args import *
from .pysat_args import *
from .setting_args import *
from .training_args import *


def get_args():
    ####################################################################################################################
    parser = argparse.ArgumentParser(
        description="Train logic gate network on the various datasets."
    )

    add_encoding_args(parser)
    add_experiment_args(parser)
    add_explainer_args(parser)
    add_model_args(parser)
    add_pysat_args(parser)
    add_settings_args(parser)
    add_training_args(parser)

    args = parser.parse_args()

    if args.verbose:
        print(vars(args))

    assert (
        args.num_iterations % args.eval_freq == 0
    ), f"iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})"

    return args


@dataclass
class DefaultArgs(
    ExperimentArgs,
    SettingsArgs,
    PySatArgs,
    ModelArgs,
    TrainingArgs,
    EncodingArgs,
    ExplainerArgs,
):
    pass
