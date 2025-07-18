from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer, classproperty


class IrisDataset(CustomDataset, AutoTransformer):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "42615765a885ddf54427f12c34a0a070",
    )

    @classproperty
    def attributes(self):
        return ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    @classproperty
    def continuous_attributes(self):
        return set(self.attributes)

    @classproperty
    def discrete_attributes(self):
        return set()  # No discrete attributes in Iris dataset

    @classproperty
    def bin_sizes(self):
        return {k: 2 for k in self.attributes}

    converter = None
    label_encoder = None

    def load_data(self):
        raw_data = self.read_raw_data(self.fpath)
        features, labels = raw_data[:, :-1], raw_data[:, -1]

        self.raw_features = features.copy()

        self.features = IrisDataset.transform_feature(features)
        self.labels = IrisDataset.transform_label(labels)
