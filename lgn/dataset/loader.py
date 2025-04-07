import torch
from .dataset import (
    load_dataset,
    IrisDataset,
    Caltech101Dataset,
    MNISTDataset,
)
from .binarizer import Binarizer


def new_load_dataset(args):
    if args.dataset == "iris":
        dataset = IrisDataset(transform=Binarizer(IrisDataset(), 2))
    elif args.dataset == "caltech101":
        dataset = Caltech101Dataset()
    elif args.dataset == "adult":
        return load_dataset(args)
    elif args.dataset in ["monk1", "monk2", "monk3"]:
        return load_dataset(args)
    elif args.dataset == "breast_cancer":
        return load_dataset(args)
    elif args.dataset == "mnist":
        dataset = MNISTDataset()

    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=int(1e6), shuffle=False
    )
    validation_loader = None
    return train_loader, validation_loader, test_loader
