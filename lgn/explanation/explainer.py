import logging

from lgn.util import feat_to_input
from lgn.encoding import Encoding
from .oracle import SatOracle

logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, encoding, Oracle=SatOracle):
        self.encoding = encoding
        self.oracle = Oracle(encoding=encoding)

    def explain(self, feat):
        logger.debug("==== Explaining: %s ====", feat)

        inp = feat_to_input(feat)
        logger.info("Explaining Input: %s", inp)

        class_label = self.encoding.as_model()(feat.reshape(1, -1))
        pred_class = class_label + 1

        logger.info("Predicted Class - %s", pred_class)

        assert self.oracle.is_uniquely_satisfied_by(inp, pred_class)

        reduced = self.reduce_input(inp, pred_class)
        logger.info("Final reduced: %s", reduced)

    def reduce_input(self, inp, predicted_cls):
        tmp_input = inp.copy()
        for feature in inp:
            logger.info("Testing removal of %d", feature)
            tmp_input.remove(feature)
            if self.oracle.is_uniquely_satisfied_by(
                inp=tmp_input, predicted_cls=predicted_cls
            ):
                continue
            else:
                tmp_input.append(feature)
        return tmp_input
