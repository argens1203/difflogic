import logging
import torch

from pysat.formula import Atom

from constant import device

fp_type = torch.float32

logger = logging.getLogger(__name__)


class PseudoModel:
    def __init__(self, input_handles, formula, class_dim):
        self.input_handles = input_handles
        self.formula = formula
        self.class_dim = class_dim

    def __call__(self, x: torch.Tensor, logit=False):
        """
        This methods returns the class label given the input. Or returns the votes for each class if logit is set to True.

        param: x: input
        type: torch.Tensor (batch_size, input_dim)

        param: logit: if True, returns votes for each class
        type: bool

        returns: class label or votes
        rtype: torch.Tensor (batch_size) or torch.Tensor (batch_size, no_of_classes)
        """
        bool_outputs = (
            torch.tensor(
                [
                    [
                        (
                            0
                            if f.simplified(
                                assumptions=[
                                    ~inp if feat == 0 else inp
                                    for feat, inp in zip(features, self.input_handles)
                                ]
                            )
                            # TODO: better way of checking?
                            == Atom(False)
                            else 1
                        )
                        for f in self.formula
                    ]
                    for features in x
                ]
            )
            .to(device)
            .int()
        )
        votes = PseudoModel._count_votes(bool_outputs, self.class_dim)

        if logit:
            return votes

        cls_label = votes.argmax().int()
        logger.debug("Class Label: %d", cls_label)
        return cls_label

    def _count_votes(logits: torch.Tensor, no_of_classes: int):
        """
        This methods takes a binary input and returns the number of 1s in each class.

        param: logits: binary input
        type: torch.Tensor (batch_size, no_of_classes * votes_per_class)

        returns: votes
        rtype: torch.Tensor (batch_size, no_of_classes)
        """
        assert logits.shape[1] % no_of_classes == 0

        votes_per_class = logits.shape[1] // no_of_classes
        reshaped = logits.view(len(logits), -1, votes_per_class)

        return reshaped.sum(dim=2)
