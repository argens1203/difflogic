from torchvision import transforms
import torchvision.datasets

from .auto_transformer import AutoTransformer
import torch


class Flatten:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1)


class MNISTDataset(AutoTransformer):
    converter = None
    label_encoder = None
    attributes = ["pixel " + str(i) for i in range(400)]
    discrete_attributes = set()
    continuous_attributes = None
    bin_sizes = {k: 2 for k in attributes}

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        feature, label = self.features[index], self.labels[index]
        return feature, label, index
