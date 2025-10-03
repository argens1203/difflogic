import os
import numpy as np
import csv
from abc import ABC, abstractmethod

from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset


class CustomDataset(Dataset, ABC):
    root = "data-uci"

    def __init__(self, transform=None, root=None):
        self.root = root if root is not None else self.root
        self.transform = transform
        self.fpath = os.path.join(self.root, self.url.split("/")[-1])

        if not check_integrity(self.fpath, self.md5):
            download_url(self.url, self.root, self.url.split("/")[-1], self.md5)
        self.load_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]

        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label, index

    def get_all(self):
        return self.features, self.labels

    def read_raw_data(self, filepath=None, delimiter=",", select=lambda x: True):
        if filepath is None:
            filepath = self.fpath

        with open(filepath, "r") as f:
            data = f.readlines()

        for i in range(len(data)):
            if len(data[i]) <= 2 or not select(data[i]):
                data[i] = None
            else:
                data[i] = data[i].strip("\n").strip().strip(".").split(delimiter)
                data[i] = [d.strip() for d in data[i]]
        data = list(filter(lambda x: x is not None, data))
        # print(data[0])
        # input("Press Enter to continue...")
        return np.array(data)

        rows = []
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(
                f, delimiter=delimiter, quotechar='"', skipinitialspace=True
            )
            for row in reader:
                # turn row into the raw line string if you need to filter before parsing
                raw_line = delimiter.join(row)
                if not row or not select(raw_line):
                    continue
                # strip spaces from each field
                cleaned = [cell.strip() for cell in row]
                rows.append(cleaned)
        print(rows[0])

        input("Press Enter to continue...")
        return np.array(rows, dtype=object)

    @abstractmethod
    def load_data(self):
        pass

    def _get_fpath(self):
        return self.fpath
