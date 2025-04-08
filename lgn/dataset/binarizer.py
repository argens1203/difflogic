from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F

import torch


class Binarizer:
    def __init__(self, dataset: Dataset, bin_count=2):
        self.bin_edges = self.get_bins(dataset, bin_count)

    def get_bins(self, dataset: Dataset, bin_count):
        feature, _ = next(
            DataLoader(dataset, batch_size=len(dataset))._get_iterator()
        )  # TODO: handle large datasets
        bins = feature.quantile(
            torch.linspace(0, 1, bin_count + 1), dim=0
        )  # len(bins) = bin_count + 1
        return bins.transpose(0, 1)  # len(bins) = feature_dim

    def __call__(self, feature):
        ret = []
        # logger.debug(f"feature={feature}")
        for f, bin_edges in zip(feature.reshape(-1, 1), self.bin_edges):
            bucket = max(torch.bucketize(f, bin_edges) - 1, torch.tensor([0]))
            ret.append(F.one_hot(bucket, num_classes=len(bin_edges) - 1))
        ret = torch.stack(ret).reshape(-1)
        # logger.debug(f"ret={ret}")
        return ret
