import torch
import torchvision.datasets
from typing import Optional

from torchvision import transforms

from .auto_transformer import AutoTransformer


class Flatten:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1)


def MNISTDatasetFactory(args):
    cls = MNISTDataset
    if args.size == "custom":
        cls.bin_size = 8
    else:
        cls.bin_size = 2
    return cls()


class MNISTDataset(AutoTransformer):
    converter = None
    label_encoder = None
    bin_size: int = 1

    @classmethod
    def attributes(cls):
        return ["pixel " + str(i) for i in range(400)]

    @classmethod
    def continuous_attributes(cls):
        return set(cls.attributes())

    @classmethod
    def discrete_attributes(cls):
        return set()

    @classmethod
    def bin_sizes(cls):
        return {k: cls.bin_size for k in cls.attributes()}

    def __init__(self):
        self.dataset = torchvision.datasets.MNIST(
            "data-uci",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((20, 20)),
                    Flatten(),
                ]
            ),
        )
        self.load_data()

    def load_data(self):
        features = transforms.Resize((20, 20)).forward(self.dataset.data)
        features, labels = (
            features.reshape(-1, 400).numpy(),
            self.dataset.targets.numpy(),
        )
        self.raw_features = features.copy()
        self.features = MNISTDataset.transform_feature(features)
        self.labels = MNISTDataset.transform_label(labels)
        # print(self.features.shape)
        # print(MNISTDataset.converter.get_attr_domains())
        # print(MNISTDataset.get_input_dim())
        # input("Press Enter to continue...")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]
        return feature, label, index
