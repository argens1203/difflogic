import numpy as np

from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer


class AdultDataset(CustomDataset, AutoTransformer):
    train_url, train_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "5d7c39d7b8804f071cdd1f2a7c460872",
    )
    test_url, test_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        "35238206dfdf7f1fe215bbb874adecdc",
    )

    converter = None
    label_encoder = None

    attributes = [
        "age",
        "workclass",
        # "fnlwgt",
        "education",
        # "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        # "label",
    ]
    continuous_attributes = {
        "age",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    }
    # discrete_attributes would be replaced in AutoTransformer to be the rest of attributes
    discrete_attributes = None
    bin_sizes = {
        "age": 5,
        "capital-gain": 3,
        "capital-loss": 3,
        "hours-per-week": 6,
    }

    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
    ):
        self.url = self.train_url if split == "train" else self.test_url
        self.md5 = self.train_md5 if split == "train" else self.test_md5

        super(AdultDataset, self).__init__(transform=transform, root=root)

    def load_data(self):
        raw_data = self.read_raw_data(
            self._get_fpath(),
            delimiter=",",
            select=lambda x: "?" not in x and "|" not in x,
        )

        # Cast as strings
        features, labels = raw_data[:, :-1].astype(str), raw_data[:, -1].astype(str)

        # Remove fnlwgt and education-num
        features = np.delete(features, [2, 4], axis=1)

        self.features = AdultDataset.transform_feature(features)
        self.labels = AdultDataset.transform_label(labels)
