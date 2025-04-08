import numpy as np

from .custom_dataset import CustomDataset

from sklearn.preprocessing import LabelEncoder
from .converter import Converter


class AdultDataset(CustomDataset):
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

        self.features = self.transform_feature(features)
        self.labels = self.transform_label(labels)

    def transform_label(self, labels):
        if AdultDataset.label_encoder is None:
            AdultDataset.label_encoder = LabelEncoder()
            AdultDataset.label_encoder.fit(labels)

        return AdultDataset.label_encoder.transform(labels)

    def transform_feature(self, features):
        if AdultDataset.converter is None:
            AdultDataset.converter = Converter(
                attributes=AdultDataset.attributes,
                continuous_attributes=AdultDataset.continuous_attributes,
                bin_sizes=AdultDataset.bin_sizes,
            )
            AdultDataset.converter.fit(features)

        return AdultDataset.converter.transform(features)

    def inverse_transform_feature(self, features):
        return AdultDataset.converter.inverse_transform(features)

    def inverse_transform_label(self, labels):
        return AdultDataset.label_encoder.inverse_transform(labels)

    # -- Getters -- #
    def get_attribute_ranges():
        return AdultDataset.converter.get_n_classes()

    def get_input_dim():
        return sum(AdultDataset.converter.get_n_classes())
