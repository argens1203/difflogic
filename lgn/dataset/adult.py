import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity
import os
from sklearn.model_selection import train_test_split
from .uci import UCIDataset
from lgn.dataset import get_attribute_ranges
from .dataset import CustomDataset

from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder

# attributes = [
#     "age",
#     "workclass",
#     "fnlwgt",
#     "education",
#     "education-num",
#     "marital-status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "capital-gain",
#     "capital-loss",
#     "hours-per-week",
#     "native-country",
# ]

# continuous_attributes = {
#     "age": ["minor", "very-young", "young", "middle-aged", "senior"],
#     # 'fnlwgt': 'Final-Weight',
#     # 'education-num': 'Education-Num',
#     "capital-gain": ["no-gain", "small-gain", "large-gain"],
#     "capital-loss": ["no-loss", "small-loss", "large-loss"],  # 5k
#     "hours-per-week": [
#         "no-hours",
#         "mini-hours",
#         "half-hours",
#         "full-hours",
#         "more-hours",
#         "most-hours",
#     ],
# }
# discrete_attributes = {
#     "workclass": [
#         "Private",
#         "Self-emp-not-inc",
#         "Self-emp-inc",
#         "Federal-gov",
#         "Local-gov",
#         "State-gov",
#         "Without-pay",
#         "Never-worked",
#     ],
#     "education": [
#         "Bachelors",
#         "Some-college",
#         "11th",
#         "HS-grad",
#         "Prof-school",
#         "Assoc-acdm",
#         "Assoc-voc",
#         "9th",
#         "7th-8th",
#         "12th",
#         "Masters",
#         "1st-4th",
#         "10th",
#         "Doctorate",
#         "5th-6th",
#         "Preschool",
#     ],
#     "marital-status": [
#         "Married-civ-spouse",
#         "Divorced",
#         "Never-married",
#         "Separated",
#         "Widowed",
#         "Married-spouse-absent",
#         "Married-AF-spouse",
#     ],
#     "occupation": [
#         "Tech-support",
#         "Craft-repair",
#         "Other-service",
#         "Sales",
#         "Exec-managerial",
#         "Prof-specialty",
#         "Handlers-cleaners",
#         "Machine-op-inspct",
#         "Adm-clerical",
#         "Farming-fishing",
#         "Transport-moving",
#         "Priv-house-serv",
#         "Protective-serv",
#         "Armed-Forces",
#     ],
#     "relationship": [
#         "Wife",
#         "Own-child",
#         "Husband",
#         "Not-in-family",
#         "Other-relative",
#         "Unmarried",
#     ],
#     "race": [
#         "White",
#         "Asian-Pac-Islander",
#         "Amer-Indian-Eskimo",
#         "Other",
#         "Black",
#     ],
#     "sex": ["Female", "Male"],
#     "native-country": [
#         "United-States",
#         "Cambodia",
#         "England",
#         "Puerto-Rico",
#         "Canada",
#         "Germany",
#         "Outlying-US(Guam-USVI-etc)",
#         "India",
#         "Japan",
#         "Greece",
#         "South",
#         "China",
#         "Cuba",
#         "Iran",
#         "Honduras",
#         "Philippines",
#         "Italy",
#         "Poland",
#         "Jamaica",
#         "Vietnam",
#         "Mexico",
#         "Portugal",
#         "Ireland",
#         "France",
#         "Dominican-Republic",
#         "Laos",
#         "Ecuador",
#         "Taiwan",
#         "Haiti",
#         "Columbia",
#         "Hungary",
#         "Guatemala",
#         "Nicaragua",
#         "Scotland",
#         "Thailand",
#         "Yugoslavia",
#         "El-Salvador",
#         "Trinadad&Tobago",
#         "Peru",
#         "Hong",
#         "Holand-Netherlands",
#     ],
# }

# discrete_attribute_options = [
#     *continuous_attributes["age"],
#     #
#     *discrete_attributes["workclass"],
#     #
#     # *continuous_attributes["fnlwgt"],
#     #
#     *discrete_attributes["education"],
#     #
#     # *continuous_attributes["education-num"],
#     #
#     *discrete_attributes["marital-status"],
#     *discrete_attributes["occupation"],
#     *discrete_attributes["relationship"],
#     *discrete_attributes["race"],
#     *discrete_attributes["sex"],
#     #
#     *continuous_attributes["capital-gain"],
#     *continuous_attributes["capital-loss"],
#     *continuous_attributes["hours-per-week"],
#     #
#     *discrete_attributes["native-country"],
# ]

# discrete_attribute_to_idx = {k: v for v, k in enumerate(discrete_attribute_options)}
# idx_to_attributes = {k: v for k, v in enumerate(discrete_attribute_options)}
# label_dict = {">50K": 1.0, "<=50K": 0.0}


class AdultDataset(CustomDataset):
    train_url, train_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "5d7c39d7b8804f071cdd1f2a7c460872",
    )
    test_url, test_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        "35238206dfdf7f1fe215bbb874adecdc",
    )

    def __init__(self, root=None, split="train", transform=None):
        self.url = self.train_url if split == "train" else self.test_url
        self.md5 = self.train_md5 if split == "train" else self.test_md5
        super(AdultDataset, self).__init__(transform=transform, root=root)

    def load_data(self):
        raw_data = self.read_raw_data(
            self._get_fpath(),
            delimiter=",",
            select=lambda x: "?" not in x and "|" not in x,
        )

        self.features, self.labels = AdultDataset.new_convert(raw_data)

    def new_convert(data):
        data = np.array(data).astype(str)
        feature, label = data[:, :-1], data[:, -1]
        feature = np.delete(feature, [2, 4], axis=1)

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
        converter = Converter(
            attributes=attributes,
            continuous_attributes=continuous_attributes,
            bin_sizes=bin_sizes,
        )
        le = LabelEncoder()

        feature = converter.fit_transform(feature)
        label = le.fit_transform(label)

        return feature, label


class Converter:
    def __init__(
        self,
        attributes,
        continuous_attributes=None,
        discrete_attributes=None,
        bin_sizes=None,
    ):
        print("converter")
        print(attributes)
        print(continuous_attributes)
        print(discrete_attributes)
        print(bin_sizes)
        self.attributes = attributes
        self.continuous_attributes = continuous_attributes
        self.discrete_attributes = discrete_attributes
        self.bin_sizes = bin_sizes

        if continuous_attributes is None:
            self.continuous_attributes = set(attributes) - discrete_attributes
        if discrete_attributes is None:
            self.discrete_attributes = set(attributes) - continuous_attributes

        self.convertors = dict()
        self.setup_convertors()

        self.ohe = None  # Note: OneHotEncoder is not initialized until fit is called
        self.n_classes = []

    def setup_convertors(self):
        for attr in self.attributes:
            if attr in self.continuous_attributes:
                self.convertors[attr] = KBinsDiscretizer(
                    n_bins=self.bin_sizes[attr], encode="ordinal", strategy="kmeans"
                )
            elif attr in self.discrete_attributes:
                self.convertors[attr] = LabelEncoder()
            else:
                raise ValueError(f"Unknown attribute {attr}")

    def transform_attr(self, data, attr):
        if attr in self.continuous_attributes:
            return (
                self.convertors[attr]
                .transform(data.astype(float).reshape(-1, 1))
                .reshape(-1)
            )
        elif attr in self.discrete_attributes:
            return self.convertors[attr].transform(data)
        else:
            raise ValueError(f"Unknown attribute {attr}")

    def fit_attr(self, data, attr):
        print("fit attr", attr)
        print(data)
        print("None" in data)
        print(np.where(data == "None"))
        converter = self.convertors[attr]

        if attr in self.continuous_attributes:
            data = data.astype(float).reshape(-1, 1)
            converter.fit(data)
            return converter.n_bins_[0].item()

        if attr in self.discrete_attributes:
            converter.fit(data)
            return len(converter.classes_)

        raise ValueError(f"Unknown attribute {attr}")

    def inverse_transform_attr(self, data, attr):
        converter = self.convertors[attr]

        if attr in self.continuous_attributes:
            data = data.reshape(-1, 1)
            return converter.inverse_transform(data).reshape(-1)

        if attr in self.discrete_attributes:
            return converter.inverse_transform(data)

        raise ValueError(f"Unknown attribute {attr}")

    def transform(self, data):
        for i, attr in enumerate(self.attributes):
            data[:, i] = self.transform_attr(data[:, i], attr)

        data = data.astype(float).astype(int)
        data = self.ohe.fit_transform(data)

        return data

    def inverse_transform(self, data):
        data = self.ohe.inverse_transform(data)
        data = data.astype(str)
        for i, attr in enumerate(self.attributes):
            data[:, i] = self.inverse_transform_attr(data[:, i].astype(int), attr)
        return data

    def fit(self, data):
        n_classes = []
        print(data[14])
        for i, attr in enumerate(self.attributes):
            n = self.fit_attr(data[:, i], attr)
            n_classes.append(n)

        print("n_classes", n_classes)
        self.ohe = OneHotEncoder(
            sparse_output=False,
        )
        self.n_classes = n_classes

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    # def new_convert(data):
    #     le = LabelEncoder()
    #     attributes = [
    #         "age",
    #         "workclass",
    #         # "fnlwgt",
    #         "education",
    #         # "education-num",
    #         "marital-status",
    #         "occupation",
    #         "relationship",
    #         "race",
    #         "sex",
    #         "capital-gain",
    #         "capital-loss",
    #         "hours-per-week",
    #         "native-country",
    #     ]
    #     continuous_attributes = {
    #         "age": ["minor", "very-young", "young", "middle-aged", "senior"],
    #         # 'fnlwgt': 'Final-Weight',
    #         # 'education-num': 'Education-Num',
    #         "capital-gain": ["no-gain", "small-gain", "large-gain"],
    #         "capital-loss": ["no-loss", "small-loss", "large-loss"],  # 5k
    #         "hours-per-week": [
    #             "no-hours",
    #             "mini-hours",
    #             "half-hours",
    #             "full-hours",
    #             "more-hours",
    #             "most-hours",
    #         ],
    #     }

    #     le_dict = dict()

    #     data = np.array(data)

    #     # print(data.shape)
    #     # print(data[0])

    #     label = data[:, -1]
    #     continuous_data = np.concat((data[:, 0], data[:, 10:13]), axis=1).astype(int)
    #     discrete_data = np.concat(
    #         (data[:, 1], data[:, 3], data[:, 5:10], data[13]), axis=1
    #     )
    #     continuous_dict = {
    #         "age": KBinsDiscretizer(n_bins=5),
    #         "capital-gain": KBinsDiscretizer(n_bins=3),
    #         "capital-loss": KBinsDiscretizer(n_bins=3),
    #         "hours-per-week": KBinsDiscretizer(n_bins=6),
    #     }
    #     print(continuous_data)
    #     print(continuous_data[0])
    #     for i, attr in enumerate(continuous_attributes):
    #         print(attr)
    #         continuous_dict[attr].fit(continuous_data[:, i])
    #     print(np.concat([cont]))

    #     data = np.concat((data[:, :2], data[:, 3], data[:, 5:-1]), axis=1)
    #     # print(data.shape)
    #     print(data[0])
    #     for i, attr in enumerate(attributes):
    #         # print(attr)
    #         if attr not in continuous_attributes:
    #             print("added to dict", attr)
    #             le_dict[attr] = LabelEncoder()
    #             le_dict[attr].fit(data[:, i])
    #             print(data[0, i])
    #             data[:, i] = le_dict[attr].transform(data[:, i])
    #             print(data[0, i])

    #     print(data[0])

    #     data = data.astype(int)
    #     # print(data.shape)

    #     # print(label.shape)
    #     # print(label[0])

    #     cp = data.copy().astype(str)
    #     for i, value in enumerate(data[0]):
    #         if attributes[i] not in continuous_attributes:
    #             print("from dict", attributes[i])
    #             cp[0, i] = le_dict[attributes[i]].inverse_transform([value])[0]
    #     print(cp[0])
    #     exit()
