from lgn.encoding import Encoding

from typing import List, Set, FrozenSet

from experiment.helpers import (
    feat_to_input,
    input_to_feat,
    Partial_Inp,
    One_Indexed_Single_Inp,
)
from lgn.dataset import AutoTransformer

from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

import numpy as np


class Instance:
    def __init__(
        self,
        feat=None,
        grouped_inp: Set[FrozenSet[int]] = None,
        raw_inp: List[int] = None,
        predicted_class=None,
        Dataset: AutoTransformer = None,
    ):
        self.feat = feat
        self.grouped_inp = grouped_inp
        self.raw_inp = raw_inp
        self.predicted_class = predicted_class

        self.Dataset = Dataset

    def get_input(self):
        return set(self.raw_inp)

    def get_input_as_set(self):
        return self.grouped_inp

    def get_predicted_class(self):
        return self.predicted_class

    def get_feature(self):
        return self.feat

    def get_attr_indices(self, p: One_Indexed_Single_Inp):
        z = p - 1

        total = 0
        idx = 0
        for step in self.Dataset.get_attribute_ranges():
            total += step
            if total > z:
                break
            idx += 1
        return idx, z - total + step

    def explain_continuous(self, offset: int, attr: str, kbd: KBinsDiscretizer):
        bin_edges = kbd.bin_edges_[0]
        if offset == 0:
            return f"{attr} smaller than {bin_edges[1]:.2f}"

        if offset == len(bin_edges) - 2:
            return f"{attr} larger than {bin_edges[-2]:.2f}"

        return f"{attr} between {bin_edges[offset - 1]:.2f} and {bin_edges[offset]:.2f}"

    def explain_discrete(self, offset: int, attr: str, le: LabelEncoder):
        return f"{attr} equal to {le.inverse_transform([offset])[0]}"

    def explain(self, p: One_Indexed_Single_Inp):
        attr_idx, attr_idx_offset = self.get_attr_indices(p)
        attr = self.Dataset.attributes()[attr_idx]
        converter = self.Dataset.converter.convertors[attr]
        if attr in self.Dataset.continuous_attributes():
            return self.explain_continuous(attr_idx_offset, attr, converter)
        else:
            return self.explain_discrete(attr_idx_offset, attr, converter)

    def verbose(self, explanation: Partial_Inp):
        positives = filter(lambda x: x > 0, explanation)
        return " AND ".join([str(self.explain(p)) for p in positives])

    # Class Methods

    @staticmethod
    def from_encoding(encoding: Encoding, feat=None, raw=None, inp=None):
        if feat is None:
            feat = encoding.get_dataset().transform_feature(np.array([raw]))[0]
        raw_inp, feat = Instance.fill_missing(inp=inp, feat=feat)

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
            Dataset=encoding.get_dataset(),
        )

    @staticmethod
    def fill_missing(inp=None, feat=None):
        if inp is None:
            inp = feat_to_input(feat)
        if feat is None:
            feat = input_to_feat(inp)
        return inp, feat
