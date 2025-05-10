import torch
from .dataset import Caltech101Dataset
from .mnist import MNISTDataset
from .binarizer import Binarizer
from .adult import AdultDataset
from .monk import Monk1Dataset, Monk2Dataset, Monk3Dataset
from .iris import IrisDataset
from .breast_cancer import BreastCancerDataset


def get_raw(dataset):
    def get_raw_data(index: int):
        return dataset.raw_features[index]

    return get_raw_data


def new_load_dataset(args):
    if args.dataset == "adult":
        train_set = AdultDataset(split="train")
        test_set = AdultDataset(split="test")
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
        return (
            train_loader,
            test_loader,
            train_set,
            test_set,
            None,
            get_raw(train_set),
            get_raw(test_set),
        )

    if args.dataset in ["monk1", "monk2", "monk3"]:
        style = int(args.dataset[4])
        assert style in [1, 2, 3], style

        if style == 1:
            train_set = Monk1Dataset()
            test_set = Monk1Dataset(split="test")
        elif style == 2:
            train_set = Monk2Dataset()
            test_set = Monk2Dataset(split="test")
        elif style == 3:
            train_set = Monk3Dataset()
            test_set = Monk3Dataset(split="test")

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
        return (
            train_loader,
            test_loader,
            train_set,
            test_set,
            None,
            get_raw(train_set),
            get_raw(test_set),
        )

    if args.dataset == "iris":
        dataset = IrisDataset()
    elif args.dataset == "breast_cancer":
        dataset = BreastCancerDataset()
    elif args.dataset == "caltech101":
        dataset = Caltech101Dataset()
    elif args.dataset == "mnist":
        dataset = MNISTDataset()

    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=int(1e6), shuffle=False
    )
    return train_loader, test_loader, get_raw(dataset)
