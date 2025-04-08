from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer


class MonkDataset(CustomDataset, AutoTransformer):
    attributes = [f"attribute_{i}" for i in range(1, 7)]
    continuous_attributes = set()  # No continuous attributes in Monk datasets
    bin_sizes = dict()  # No bin sizes since no continuous attributes

    discrete_attributes = set(attributes)
    converter = None
    label_encoder = None

    def __init__(self, root=None, split="train", transform=None):
        self.url = self.train_url if split == "train" else self.test_url
        self.md5 = self.train_md5 if split == "train" else self.test_md5

        super(MonkDataset, self).__init__(transform=transform, root=root)

    def load_data(self):
        raw_data = self.read_raw_data(self._get_fpath(), delimiter=" ")
        raw_data = raw_data[:, :-1]  # Remove instance id
        labels, features = raw_data[:, 0], raw_data[:, 1:]

        self.raw_features = features.copy()

        self.features = MonkDataset.transform_feature(features)
        self.labels = MonkDataset.transform_label(labels)


class Monk1Dataset(MonkDataset):
    test_url, test_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test",
        "de4255acb72fb29be5125a7c874e28a0",
    )
    train_url, train_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train",
        "fc1fc3a673e00908325c67cf16283335",
    )


class Monk2Dataset(MonkDataset):
    train_url, train_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train",
        "f109ee3f95805745af6cdff06d6fbc94",
    )
    test_url, test_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test",
        "106cb9049ba5ccd7969a0bd5ff19681d",
    )


class Monk3Dataset(MonkDataset):
    train_url, train_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train",
        "613e44dbb8ffdf54d364bd91e4e74afd",
    )

    test_url, test_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test",
        "46815731e31c07f89422cf60de8738e7",
    )
