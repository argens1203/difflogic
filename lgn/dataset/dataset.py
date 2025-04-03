import logging
import experiments.mnist_dataset as mnist_dataset
import experiments.uci_datasets as uci_datasets
import torch
import math
import torchvision
import torchvision.transforms as T

logger = logging.getLogger(__name__)


def load_dataset(args):
    validation_loader = None
    if args.dataset == "iris":
        train_set = uci_datasets.IrisDataset(
            "./data-uci", split="train", download=True, with_val=False
        )
        test_set = uci_datasets.IrisDataset("./data-uci", split="test", with_val=False)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
    elif args.dataset == "adult":
        train_set = uci_datasets.AdultDataset(
            "./data-uci", split="train", download=True, with_val=False
        )
        test_set = uci_datasets.AdultDataset("./data-uci", split="test", with_val=False)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
    elif args.dataset == "breast_cancer":
        train_set = uci_datasets.BreastCancerDataset(
            "./data-uci", split="train", download=True, with_val=False
        )
        test_set = uci_datasets.BreastCancerDataset(
            "./data-uci", split="test", with_val=False
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
    elif args.dataset.startswith("monk"):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset(
            "./data-uci", style, split="train", download=True, with_val=False
        )
        test_set = uci_datasets.MONKsDataset(
            "./data-uci", style, split="test", with_val=False
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
    elif args.dataset in ["mnist", "mnist20x20"]:
        train_set = mnist_dataset.MNIST(
            "./data-mnist",
            train=True,
            download=True,
            remove_border=args.dataset == "mnist20x20",
            transform=T.Compose([T.Lambda(lambda x: torch.flatten(x))]),
        )
        test_set = mnist_dataset.MNIST(
            "./data-mnist",
            train=False,
            remove_border=args.dataset == "mnist20x20",
            transform=T.Compose([T.Lambda(lambda x: torch.flatten(x))]),
        )

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(
            train_set, [train_set_size, valid_set_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
    elif "cifar-10" in args.dataset:
        transform = {
            "cifar-10-3-thresholds": lambda x: torch.cat(
                [(x > (i + 1) / 4).float() for i in range(3)], dim=0
            ),
            "cifar-10-31-thresholds": lambda x: torch.cat(
                [(x > (i + 1) / 32).float() for i in range(31)], dim=0
            ),
        }[args.dataset]
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(transform),
            ]
        )
        train_set = torchvision.datasets.CIFAR10(
            "./data-cifar", train=True, download=True, transform=transforms
        )
        test_set = torchvision.datasets.CIFAR10(
            "./data-cifar", train=False, transform=transforms
        )

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(
            train_set, [train_set_size, valid_set_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

    else:
        raise NotImplementedError(f"The data set {args.dataset} is not supported!")

    return train_loader, validation_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def input_dim_of_dataset(dataset):  # TODO: get it from Dataset class
    return {
        "adult": 116,
        "breast_cancer": 51,
        "iris": 4 * 2,
        "monk1": 17,
        "monk2": 17,
        "monk3": 17,
        "mnist": 400 * 2,
        "mnist20x20": 400,
        "cifar-10-3-thresholds": 3 * 32 * 32 * 3,
        "cifar-10-31-thresholds": 3 * 32 * 32 * 31,
        "caltech101": 64 * 64 * 2,
    }[dataset]


def num_classes_of_dataset(dataset):  # TODO: get it from Dataset class
    return {
        "adult": 2,
        "breast_cancer": 2,
        "iris": 3,
        "monk1": 2,
        "monk2": 2,
        "monk3": 2,
        "mnist": 10,
        "mnist20x20": 10,
        "cifar-10-3-thresholds": 10,
        "cifar-10-31-thresholds": 10,
        "caltech101": 101,
    }[dataset]


def get_attribute_ranges(dataset):
    return {
        "monk1": [3, 3, 2, 3, 4, 2],
        "monk2": [3, 3, 2, 3, 4, 2],
        "monk3": [3, 3, 2, 3, 4, 2],
        "iris": [2, 2, 2, 2],
    }[dataset]


from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset, DataLoader
import os


class CustomDataset(Dataset):
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
        return feature, label

    def get_all(self):
        return self.features, self.labels

    def read_raw_data(self, filepath):
        with open(filepath, "r") as f:
            data = f.readlines()

        for i in range(len(data)):
            if len(data[i]) <= 2:
                data[i] = None
            else:
                data[i] = data[i].strip("\n").strip().split(",")
                data[i] = [d.strip() for d in data[i]]

        data = list(filter(lambda x: x is not None, data))
        return data


class IrisDataset(CustomDataset):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "42615765a885ddf54427f12c34a0a070",
    )
    # location = "iris.data"

    label_dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

    def load_data(self):
        def parse_feature(features):
            return [float(f) for f in features]

        def parse(data):
            features = [parse_feature(sample[:-1]) for sample in data]
            labels = [self.label_dict[sample[-1]] for sample in data]
            features, labels = (
                torch.tensor(features).float(),
                torch.tensor(labels).float(),
            )

            return features, labels

        raw_data = self.read_raw_data(self.fpath)
        self.features, self.labels = parse(raw_data)


class AdultDataset(CustomDataset):
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
    label_dict = {"<=50K": 0, ">50K": 1}

    def __init__(self, train=True, transform=None):
        self.url, self.md5 = self.file_list[0] if train else self.file_list[1]
        CustomDataset.__init__(self, transform)

    def load_data(self):
        def parse_feature(features):
            print(features)
            exit()
            return [float(f) for f in features]

        def parse(data):
            features = [parse_feature(sample[:-1]) for sample in data]
            labels = [self.label_dict[sample[-1]] for sample in data]
            features, labels = (
                torch.tensor(features).float(),
                torch.tensor(labels).float(),
            )

            return features, labels

        raw_data = self.read_raw_data(self.fpath)
        self.features, self.labels = parse(raw_data)


class MonkDataset(CustomDataset):
    file_list = [
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train",
            "fc1fc3a673e00908325c67cf16283335",
        ),
        (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test",
            "de4255acb72fb29be5125a7c874e28a0",
        ),
    ]

    def __init__(self, train=True, transform=None):
        self.url, self.md5 = self.file_list[0] if train else self.file_list[1]
        CustomDataset.__init__(self, transform)


class BreastCancerDataset(CustomDataset):
    url, md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data",
        "d887db31e2b99e39c4b7fc79d2f36a3a",
    )

    def __init__(self, train=True, transform=None):
        CustomDataset.__init__(self, transform)


import torch.nn.functional as F


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


class Flatten:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


import torchvision.datasets
from torchvision import transforms


class Caltech101Dataset:
    dataset = torchvision.datasets.Caltech101(
        "data-uci",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.Grayscale(),
                Flatten(),
            ]
        ),
    )
    dataset = torchvision.datasets.Caltech101(
        "data-uci",
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
                transforms.Grayscale(),
                Flatten(),
                Binarizer(dataset, 2),
            ]
        ),
    )

    def __call__(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class MNISTDataset:
    dataset = torchvision.datasets.MNIST(
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
    dataset = torchvision.datasets.MNIST(
        "data-uci",
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((20, 20)),
                Flatten(),
                Binarizer(dataset, 2),
            ]
        ),
    )

    def __call__(self):
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
