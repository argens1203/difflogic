import torch
from .custom_dataset import CustomDataset


class IrisDataset(CustomDataset):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "42615765a885ddf54427f12c34a0a070",
    )
    # location = "iris.data"

    label_dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    def load_data(self):
        def parse_feature(features):
            return [float(f) for f in features]

        def parse(data):
            features = [parse_feature(sample[:-1]) for sample in data]
            labels = [self.label_dict[sample[-1]] for sample in data]
            features, labels = (
                torch.tensor(features).float(),
                torch.tensor(labels).float(),
            )

            return features, labels

        raw_data = self.read_raw_data(self.fpath)
        self.features, self.labels = parse(raw_data)
