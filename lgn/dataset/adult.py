import numpy as np

from .dataset import CustomDataset

from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder
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
        converter=None,
        label_encoder=None,
    ):
        self.url = self.train_url if split == "train" else self.test_url
        self.md5 = self.train_md5 if split == "train" else self.test_md5
        self.converter = converter
        self.label_encoder = label_encoder

        super(AdultDataset, self).__init__(transform=transform, root=root)

    def load_data(self):
        raw_data = self.read_raw_data(
            self._get_fpath(),
            delimiter=",",
            select=lambda x: "?" not in x and "|" not in x,
        )

        self.features, self.labels = self.binarize(raw_data)

    def binarize(self, data):
        data = np.array(data).astype(str)
        feature, label = data[:, :-1], data[:, -1]
        feature = np.delete(feature, [2, 4], axis=1)

        if self.converter is None:
            self.converter = Converter(
                attributes=AdultDataset.attributes,
                continuous_attributes=AdultDataset.continuous_attributes,
                bin_sizes=AdultDataset.bin_sizes,
            )
            self.converter.fit(feature)

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(label)

        feature = self.converter.transform(feature)
        label = self.label_encoder.transform(label)

        return feature, label
