import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, check_integrity
import os
from sklearn.model_selection import train_test_split

from .uci import UCIDataset


class MONKsDataset(UCIDataset):
    file_list = [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train",
            "fc1fc3a673e00908325c67cf16283335",
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test",
            "de4255acb72fb29be5125a7c874e28a0",
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train",
            "f109ee3f95805745af6cdff06d6fbc94",
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test",
            "106cb9049ba5ccd7969a0bd5ff19681d",
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train",
            "613e44dbb8ffdf54d364bd91e4e74afd",
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test",
            "46815731e31c07f89422cf60de8738e7",
        ),
    ]

    def __init__(self, root, style: int, split="train", download=False, with_val=False):
        super(MONKsDataset, self).__init__(root, split, download)
        self.style = style
        assert style in [1, 2, 3], style

        if self.split == "val":
            assert with_val

        if self.split in ["train", "val"]:
            self.data, self.labels = MONKsDataset.preprocess_monks_to_binary_data(
                os.path.join(root, "monks-{}.train".format(style))
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
            self.data, self.labels = MONKsDataset.preprocess_monks_to_binary_data(
                os.path.join(root, "monks-{}.test".format(style))
            )

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        return data, label

    @staticmethod
    def preprocess_monks_to_binary_data(data_file_name):

        def read_raw_data(filepath):
            with open(filepath, "r") as f:
                data = f.readlines()

            for i in range(len(data)):
                if len(data[i]) <= 2:
                    data[i] = None
                else:
                    data[i] = data[i].strip("\n").strip(".").strip().split(" ")
                    data[i] = [d for d in data[i]]
                    data[i] = data[i][:-1]

            data = list(filter(lambda x: x is not None, data))

            return data

        def convert_sample_to_feature_vector(sample):
            attribute_ranges = [3, 3, 2, 3, 4, 2]

            D = sum(attribute_ranges)
            vec = np.zeros(D)

            count = 0
            for i, attribute_range in enumerate(attribute_ranges):
                val = int(sample[i + 1]) - 1
                vec[count + val] = 1

                count += attribute_range

            return vec

        def convert_data_to_feature_vectors(data):
            feat = [convert_sample_to_feature_vector(sample) for sample in data]
            return torch.tensor(feat).float()

        def get_labels(data):
            num_samples = len(data)
            labels = np.zeros(num_samples, dtype=np.float32)
            for i, sample in enumerate(data):
                labels[i] = int(sample[0])

            return torch.tensor(labels).long()

        data = read_raw_data(data_file_name)

        feat = convert_data_to_feature_vectors(data)
        labels = get_labels(data)

        return feat, labels
