from lgn.util import feat_to_input, input_to_feat
from lgn.encoding import Encoding


class Instance:
    def __init__(self, feat=None, inp=None, predicted_class=None):
        self.feat = feat
        self.inp = inp
        self.predicted_class = predicted_class

    def get_input(self):
        return self.inp

    def get_predicted_class(self):
        return self.predicted_class

    def get_feature(self):
        return self.feat

    # Class Methods

    def from_encoding(encoding: Encoding, feat=None, inp=None):
        inp, feat = Instance.fill_missing(inp=inp, feat=feat)
        class_label = encoding.as_model()(feat.reshape(1, -1)).item()
        pred_class = class_label + 1
        return Instance(feat=feat, inp=inp, predicted_class=pred_class)

    def fill_missing(inp=None, feat=None):
        if inp is None:
            inp = feat_to_input(feat)
        if feat is None:
            feat = input_to_feat(inp)
        return inp, feat
