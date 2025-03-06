import experiments.mnist_dataset as mnist_dataset
import experiments.uci_datasets as uci_datasets
import torch
import math
import torchvision


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
        )
        test_set = mnist_dataset.MNIST(
            "./data-mnist", train=False, remove_border=args.dataset == "mnist20x20"
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


def input_dim_of_dataset(dataset):
    return {
        "adult": 116,
        "breast_cancer": 51,
        "iris": 8,
        "monk1": 17,
        "monk2": 17,
        "monk3": 17,
        "mnist": 784,
        "mnist20x20": 400,
        "cifar-10-3-thresholds": 3 * 32 * 32 * 3,
        "cifar-10-31-thresholds": 3 * 32 * 32 * 31,
    }[dataset]


def num_classes_of_dataset(dataset):
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
    }[dataset]
