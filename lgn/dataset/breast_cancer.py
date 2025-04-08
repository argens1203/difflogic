import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity
import os
from sklearn.model_selection import train_test_split

from .uci import UCIDataset


class BreastCancerDataset(UCIDataset):
    file_list = [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data",
            "d887db31e2b99e39c4b7fc79d2f36a3a",
        ),
    ]

    def __init__(self, root, split="train", download=False, with_val=False):
        super(BreastCancerDataset, self).__init__(root, split, download)

        if self.split == "val":
            assert with_val

        self.data, self.labels = BreastCancerDataset.preprocess_data(
            os.path.join(root, "breast-cancer.data")
        )
        data_train, data_test, labels_train, labels_test = train_test_split(
            self.data, self.labels, test_size=0.25, random_state=0
        )

        if self.split in ["train", "val"]:
            self.data, self.labels = data_train, labels_train
            if with_val:
                data_train, data_val, labels_train, labels_val = train_test_split(
                    self.data, self.labels, test_size=0.2, random_state=0
                )
                if split == "train":
                    self.data, self.labels = data_train, labels_train
                elif split == "val":
                    self.data, self.labels = data_val, labels_val
                else:
                    raise ValueError(split)
        else:
            self.data, self.labels = data_test, labels_test

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        return data, label

    @staticmethod
    def preprocess_data(data_file_name):

        attributes = {
            "age": [
                "10-19",
                "20-29",
                "30-39",
                "40-49",
                "50-59",
                "60-69",
                "70-79",
                "80-89",
                "90-99",
            ],
            "menopause": ["lt40", "ge40", "premeno"],
            "tumor-size": [
                "0-4",
                "5-9",
                "10-14",
                "15-19",
                "20-24",
                "25-29",
                "30-34",
                "35-39",
                "40-44",
                "45-49",
                "50-54",
                "55-59",
            ],
            "inv-nodes": [
                "0-2",
                "3-5",
                "6-8",
                "9-11",
                "12-14",
                "15-17",
                "18-20",
                "21-23",
                "24-26",
                "27-29",
                "30-32",
                "33-35",
                "36-39",
            ],
            "node-caps": ["yes", "no"],
            "deg-malig": ["1", "2", "3"],
            "breast": ["left", "right"],
            "breast-quad": ["left_up", "left_low", "right_up", "right_low", "central"],
            "irradiat": ["yes", "no"],
        }

        label_dict = {"no-recurrence-events": 0, "recurrence-events": 1}

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
            D = sum([len(attributes[key]) for key in attributes])
            vec = np.zeros(D)

            count = 0
            for i, key in enumerate(attributes):
                val = attributes[key].index(sample[i + 1])
                vec[count + val] = 1

                count += len(attributes[key])

            return vec

        def convert_data_to_feature_vectors(data):
            feat = [convert_sample_to_feature_vector(sample) for sample in data]
            return torch.tensor(feat).float()

        def get_labels(data):
            num_samples = len(data)
            labels = np.zeros(num_samples, dtype=np.float32)
            for i, sample in enumerate(data):
                labels[i] = label_dict[sample[0]]

            return torch.tensor(labels).long()

        data = read_raw_data(data_file_name)
        data = discard_missing_data(data)

        feat = convert_data_to_feature_vectors(data)
        labels = get_labels(data)

        return feat, labels
