import torch
import logging

from constant import device
from ..util import get_truth_table_loader

logger = logging.getLogger(__name__)


class Validator:
    def validate(self, model, data=None):
        if data != None:
            Validator.validate_with_data(encoding=self, model=model, data=data)
        Validator.validate_with_truth_table(encoding=self, model=model)

    def validate_with_data(encoding, model, data):
        with torch.no_grad(), encoding.use_context():
            model.train(False)
            logger.debug("Checking model with data")
            for x, _ in data:
                x = x.to(encoding.fp_type).to(device)

                logit = model(x)
                p_logit = encoding.predict_votes(x)
                assert logit.equal(p_logit)

    def validate_with_truth_table(encoding, model):
        logger.debug("Checking model with truth table")
        with torch.no_grad(), encoding.use_context():
            model.train(False)

            for x, _ in get_truth_table_loader(input_dim=encoding.input_dim):
                x = x.to(encoding.fp_type).to(device)

                logit = model(x)
                p_logit = encoding.predict_votes(x)
                assert logit.equal(p_logit)
