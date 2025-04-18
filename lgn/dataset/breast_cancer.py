from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer


class BreastCancerDataset(CustomDataset, AutoTransformer):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data",
        "d887db31e2b99e39c4b7fc79d2f36a3a",
    )
    converter = None
    label_encoder = None

    attributes = [
        "age",
        "menopause",
        "tumor-size",
        "inv-nodes",
        "node-caps",
        "deg-malig",
        "breast",
        "breast-quad",
        "irradiat",
    ]
    continuous_attributes = set()
    discrete_attributes = set(attributes)
    bin_sizes = dict()

    def load_data(self):
        raw_data = self.read_raw_data(select=lambda x: "?" not in x)
        labels, features = raw_data[:, 0], raw_data[:, 1:]

        self.raw_features = features.copy()

        self.features = BreastCancerDataset.transform_feature(features)
        self.labels = BreastCancerDataset.transform_label(labels)
