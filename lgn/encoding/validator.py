import torch
import logging

from constant import device
from lgn.dataset.auto_transformer import AutoTransformer
from lgn.dataset.custom_dataset import CustomDataset
from lgn.encoding import Encoding
from experiment.helpers import get_truth_table_loader, get_onehot_loader, Context

logger = logging.getLogger(__name__)


class Validator:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def validate(self, encoding, model, data=None):
        if data != None:
            self.validate_with_data(encoding=encoding, model=model, data=data)
        else:
            self.validate_with_truth_table(encoding=encoding, model=model)

    def validate_with_data(self, encoding: Encoding, model, data):
        """
        The method ensures that the encoding is correct by comparing the output of the encoding with that of the model
        for all instances in a dataset

        Example:

        .. code-block:: python
            >>> Validator.validate_with_truth_table(encoding=encoding, model=model)
        """
        logger.info("Checking model with data...")

        with torch.no_grad():
            model.train(False)
            for x, _ in data:
                x = x.to(self.ctx.get_fp_type()).to(device)

                logit = model(x)
                p_logit = encoding.as_model()(x, logit=True)
                assert logit.equal(p_logit)

    def validate_encodings_with_data(
        self, encoding1: Encoding, encoding2: Encoding, dataloader
    ):
        logger.info("Checking encodings with data...")

        with torch.no_grad():
            for x, label, idx in dataloader:
                x = x.to(self.ctx.get_fp_type()).to(device)
                logit1 = encoding1.as_model()(x, logit=True)
                logit2 = encoding2.as_model()(x, logit=True)

                assert logit1.equal(logit2), (
                    f"Logits are not equal for input {idx}:\n"
                    f"Encoding 1: {logit1}\n"
                    f"Encoding 2: {logit2}"
                )

    def validate_with_truth_table(self, encoding: Encoding, model: torch.nn.Module):
        """
        The method ensures that the encoding is correct by comparing the output of the encoding with that of the model
        for all possible inputs.

        Example:

        .. code-block:: python
            >>> Validator.validate_with_truth_table(encoding=encoding, model=model)
        """
        logger.info("Checking model with truth table...")

        with torch.no_grad():
            model.train(False)

            for x, _ in get_truth_table_loader(input_dim=encoding.get_input_dim()):
                x = x.to(self.ctx.get_fp_type()).to(device)

                logit = model(x)
                p_logit = encoding.as_model()(x, logit=True)
                assert logit.equal(p_logit)

    def validate_encodings_with_truth_table(
        self, encoding1: Encoding, encoding2: Encoding, dataset: AutoTransformer
    ):
        logger.info("Checking encodings with truth table...")
        for x, _ in get_onehot_loader(
            input_dim=dataset.get_input_dim(),
            attribute_ranges=dataset.get_attribute_ranges(),
        ):
            x = x.to(self.ctx.get_fp_type()).to(device)
            logit1 = encoding1.as_model()(x, logit=True)
            logit2 = encoding2.as_model()(x, logit=True)

            assert logit1.equal(logit2), (
                f"Logits are not equal for input {x}:\n"
                f"Encoding 1: {logit1}\n"
                f"Encoding 2: {logit2}"
            )
