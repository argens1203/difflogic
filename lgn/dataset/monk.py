from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer

from abc import ABC, abstractmethod


class MonkDataset(CustomDataset, AutoTransformer, ABC):
    @classmethod
    def attributes(cls):
        return [f"attribute_{i}" for i in range(1, 7)]

    @classmethod
    def continuous_attributes(cls):
        return set()

    @classmethod
    def discrete_attributes(cls):
        return set(cls.attributes())

    @classmethod
    def bin_sizes(cls):
        return dict()

    @classmethod
    @abstractmethod
    def train_url(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def test_url(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def train_md5(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def test_md5(cls) -> str:
        pass

    converter = None
    label_encoder = None

    def __init__(self, root=None, split="train", transform=None):
        self.url: str = self.train_url() if split == "train" else self.test_url()
        self.md5: str = self.train_md5() if split == "train" else self.test_md5()

        super(MonkDataset, self).__init__(transform=transform, root=root)

    def load_data(self):
        raw_data = self.read_raw_data(self._get_fpath(), delimiter=" ")
        raw_data = raw_data[:, :-1]  # Remove instance id
        labels, features = raw_data[:, 0], raw_data[:, 1:]

        self.raw_features = features.copy()

        self.features = MonkDataset.transform_feature(features)
        self.labels = MonkDataset.transform_label(labels)


class Monk1Dataset(MonkDataset):
    @classmethod
    def test_url(cls):
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test"

    @classmethod
    def train_url(cls):
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train"

    @classmethod
    def test_md5(cls):
        return "de4255acb72fb29be5125a7c874e28a0"

    @classmethod
    def train_md5(cls):
        return "fc1fc3a673e00908325c67cf16283335"


class Monk2Dataset(MonkDataset):
    @classmethod
    def test_url(cls):
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test"

    @classmethod
    def train_url(cls):
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train"

    @classmethod
    def test_md5(cls):
        return "106cb9049ba5ccd7969a0bd5ff19681d"

    @classmethod
    def train_md5(cls):
        return "f109ee3f95805745af6cdff06d6fbc94"


class Monk3Dataset(MonkDataset):
    @classmethod
    def train_url(cls):
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train"

    @classmethod
    def train_md5(cls):
        return "613e44dbb8ffdf54d364bd91e4e74afd"

    @classmethod
    def test_url(cls):
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test"

    @classmethod
    def test_md5(cls):
        return "46815731e31c07f89422cf60de8738e7"
