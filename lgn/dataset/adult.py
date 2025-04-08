import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity
import os
from sklearn.model_selection import train_test_split
from .uci import UCIDataset
from lgn.dataset import get_attribute_ranges

attributes = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

continuous_attributes = {
    "age": ["minor", "very-young", "young", "middle-aged", "senior"],
    # 'fnlwgt': 'Final-Weight',
    # 'education-num': 'Education-Num',
    "capital-gain": ["no-gain", "small-gain", "large-gain"],
    "capital-loss": ["no-loss", "small-loss", "large-loss"],  # 5k
    "hours-per-week": [
        "no-hours",
        "mini-hours",
        "half-hours",
        "full-hours",
        "more-hours",
        "most-hours",
    ],
}
discrete_attributes = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    ],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ],
}

discrete_attribute_options = [
    *continuous_attributes["age"],
    #
    *discrete_attributes["workclass"],
    #
    # *continuous_attributes["fnlwgt"],
    #
    *discrete_attributes["education"],
    #
    # *continuous_attributes["education-num"],
    #
    *discrete_attributes["marital-status"],
    *discrete_attributes["occupation"],
    *discrete_attributes["relationship"],
    *discrete_attributes["race"],
    *discrete_attributes["sex"],
    #
    *continuous_attributes["capital-gain"],
    *continuous_attributes["capital-loss"],
    *continuous_attributes["hours-per-week"],
    #
    *discrete_attributes["native-country"],
]

discrete_attribute_to_idx = {k: v for v, k in enumerate(discrete_attribute_options)}
idx_to_attributes = {k: v for k, v in enumerate(discrete_attribute_options)}
label_dict = {">50K": 1.0, "<=50K": 0.0}


class AdultDataset(UCIDataset):
    file_list = [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "5d7c39d7b8804f071cdd1f2a7c460872",
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
            "35238206dfdf7f1fe215bbb874adecdc",
        ),
    ]

    def __init__(self, root, split="train", download=False, with_val=True):
        super(AdultDataset, self).__init__(root, split, download)

        if self.split == "val":
            assert with_val

        if self.split in ["train", "val"]:
            self.data, self.labels = AdultDataset.preprocess_adult_to_binary_data(
                os.path.join(root, "adult.data")
            )
            if with_val:
                data_train, data_val, labels_train, labels_val = train_test_split(
                    self.data, self.labels, test_size=0.1, random_state=0
                )
                if split == "train":
                    self.data, self.labels = data_train, labels_train
                elif split == "val":
                    self.data, self.labels = data_val, labels_val
                else:
                    raise ValueError(split)
        else:
            self.data, self.labels = AdultDataset.preprocess_adult_to_binary_data(
                os.path.join(root, "adult.test")
            )

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        return data, label

    @staticmethod
    def preprocess_adult_to_binary_data(data_file_name):

        def read_raw_data(filepath):
            with open(filepath, "r") as f:
                data = f.readlines()

            for i in range(len(data)):
                if data[i].startswith("|") or len(data[i]) <= 2:
                    data[i] = None
                else:
                    data[i] = data[i].strip("\n").strip(".").strip().split(",")
                    data[i] = [d.strip() for d in data[i]]

            data = list(filter(lambda x: x is not None, data))

            return data

        def discard_missing_data(data):
            num_samples = len(data)
            for i in range(num_samples):
                if "?" in data[i]:
                    data[i] = None

            data = [sample for sample in data if sample is not None]
            return data

        def convert_sample_to_feature_vector(sample):
            D = len(discrete_attribute_options)
            vec = np.zeros(D)
            for i, attr_type in enumerate(attributes):
                # 14 is skipped as it is the label
                if attr_type in ["education-num", "fnlwgt"]:
                    continue
                attr_value = sample[i]
                if attr_type in continuous_attributes:
                    attr_v = None
                    attr_value = int(attr_value)
                    if attr_type == "age":
                        if attr_value < 21:
                            attr_v = "minor"
                        elif attr_value < 30:
                            attr_v = "very-young"
                        elif attr_value < 45:
                            attr_v = "young"
                        elif attr_value < 60:
                            attr_v = "middle-aged"
                        else:
                            attr_v = "senior"
                    elif attr_type == "capital-gain":
                        if attr_value == 0:
                            attr_v = "no-gain"
                        elif attr_value < 5_000:
                            attr_v = "small-gain"
                        else:
                            attr_v = "large-gain"
                    elif attr_type == "capital-loss":
                        if attr_value == 0:
                            attr_v = "no-loss"
                        elif attr_value < 5_000:
                            attr_v = "small-loss"
                        else:
                            attr_v = "large-loss"
                    elif attr_type == "hours-per-week":
                        if attr_value == 0:
                            attr_v = "no-hours"
                        elif attr_value <= 12:
                            attr_v = "mini-hours"
                        elif attr_value <= 25:
                            attr_v = "half-hours"
                        elif attr_value <= 40:
                            attr_v = "full-hours"
                        elif attr_value < 60:
                            attr_v = "more-hours"
                        else:
                            attr_v = "most-hours"
                    else:
                        raise ValueError(attr_type)

                    vec_idx = discrete_attribute_to_idx[attr_v]
                else:
                    vec_idx = discrete_attribute_to_idx[attr_value]
                vec[vec_idx] = 1

            idx = 0
            for step in get_attribute_ranges("adult"):
                assert sum(vec[idx : idx + step]) == 1, (
                    f"feat: {vec[idx : idx + step]}" + f"sample: {sample}"
                )
                idx += step
            return vec

        def convert_data_to_feature_vectors(data):
            feat = [convert_sample_to_feature_vector(sample) for sample in data]
            return torch.tensor(feat).float()

        def get_labels(data):
            num_samples = len(data)
            labels = np.zeros(num_samples, dtype=np.float32)
            for i, sample in enumerate(data):
                labels[i] = label_dict[sample[-1]]

            return torch.tensor(labels).long()

        data = read_raw_data(data_file_name)
        data = discard_missing_data(data)

        feat = convert_data_to_feature_vectors(data)
        labels = get_labels(data)

        return feat, labels

    def translate(inp):
        """
        Translate the input to the original feature space.
        """
        return [idx_to_attributes[abs(i) - 1] for i in inp]
