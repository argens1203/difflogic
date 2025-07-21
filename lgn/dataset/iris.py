from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer


class IrisDataset(CustomDataset, AutoTransformer):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "42615765a885ddf54427f12c34a0a070",
    )

    @classmethod
    def attributes(cls):
        return ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    @classmethod
    def continuous_attributes(cls):
        return set(cls.attributes())

    @classmethod
    def discrete_attributes(cls):
        return set()  # No discrete attributes in Iris dataset

    @classmethod
    def bin_sizes(cls):
        return {k: 2 for k in cls.attributes()}

    converter = None
    label_encoder = None

    def load_data(self):
        raw_data = self.read_raw_data(self.fpath)
        features, labels = raw_data[:, :-1], raw_data[:, -1]

        self.raw_features = features.copy()

        self.features = IrisDataset.transform_feature(features)
        self.labels = IrisDataset.transform_label(labels)
