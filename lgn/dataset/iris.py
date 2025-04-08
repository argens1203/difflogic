from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer


class IrisDataset(CustomDataset, AutoTransformer):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "42615765a885ddf54427f12c34a0a070",
    )

    attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    continuous_attributes = set(attributes)
    discrete_attributes = set()  # No discrete attributes in Iris dataset
    bin_sizes = {k: 2 for k in attributes}

    converter = None
    label_encoder = None

    def load_data(self):
        raw_data = self.read_raw_data(self.fpath)
        features, labels = raw_data[:, :-1], raw_data[:, -1]

        self.features = IrisDataset.transform_feature(features)
        self.labels = IrisDataset.transform_label(labels)
