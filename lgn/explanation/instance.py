from typing import List, Set, FrozenSet

from lgn.util import feat_to_input, input_to_feat
from lgn.encoding import Encoding


class Instance:
    def __init__(
        self,
        feat=None,
        grouped_inp: Set[FrozenSet[int]] = None,
        raw_inp: List[int] = None,
        predicted_class=None,
    ):
        self.feat = feat
        self.grouped_inp = grouped_inp
        self.raw_inp = raw_inp
        self.predicted_class = predicted_class

    def get_input(self):
        return set(self.raw_inp)

    def get_input_as_set(self):
        return self.grouped_inp

    def get_predicted_class(self):
        return self.predicted_class

    def get_feature(self):
        return self.feat

    # Class Methods

    def from_encoding(encoding: Encoding, feat=None, grouped_inp=None):
        raw_inp, feat = Instance.fill_missing(inp=grouped_inp, feat=feat)

        grouped_inp = set()
        idx = 0
        for step in encoding.get_attribute_ranges():
            grouped_inp.add(frozenset(raw_inp[idx : idx + step]))
            idx += step

        class_label = encoding.as_model()(feat.reshape(1, -1)).item()
        pred_class = class_label + 1
        return Instance(
            feat=feat,
            grouped_inp=grouped_inp,
            raw_inp=raw_inp,
            predicted_class=pred_class,
        )

    def fill_missing(inp=None, feat=None):
        if inp is None:
            inp = feat_to_input(feat)
        if feat is None:
            feat = input_to_feat(inp)
        return inp, feat
