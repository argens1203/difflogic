import torch

from .custom_dataset import CustomDataset


class MonkDataset(CustomDataset):
    attribute_ranges = [3, 3, 2, 3, 4, 2]

    def __init__(self, root=None, split="train", transform=None):
        self.url = self.train_url if split == "train" else self.test_url
        self.md5 = self.train_md5 if split == "train" else self.test_md5

        super(MonkDataset, self).__init__(transform=transform, root=root)

    def load_data(self):

        def to_one_hot(features):
            # -1 since F.one_hot expects 0-indexed values
            features = torch.tensor(features) - 1

            # one_hot_features = torch.nn.functional.one_hot(features, num_classes=-1)
            # This take the largest value + 1 in all dimension as number of classes, which does not work for our uneven attribute ranges

            return torch.cat(
                [
                    torch.nn.functional.one_hot(features[:, i], num_classes=-1)
                    for i in range(len(self.attribute_ranges))
                ],
                dim=1,
            )

        def parse(sample):
            # Remove instance id
            sample = sample[:, :-1]

            # First column is the label
            features, labels = sample[:, 1:].astype(int), sample[:, 0].astype(int)

            return to_one_hot(features), torch.tensor(labels)

        raw_data = self.read_raw_data(self._get_fpath(), delimiter=" ")
        self.features, self.labels = parse(raw_data)


class Monk1Dataset(MonkDataset):
    test_url, test_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test",
        "de4255acb72fb29be5125a7c874e28a0",
    )
    train_url, train_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train",
        "fc1fc3a673e00908325c67cf16283335",
    )


class Monk2Dataset(MonkDataset):
    train_url, train_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.train",
        "f109ee3f95805745af6cdff06d6fbc94",
    )
    test_url, test_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-2.test",
        "106cb9049ba5ccd7969a0bd5ff19681d",
    )


class Monk3Dataset(MonkDataset):
    train_url, train_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train",
        "613e44dbb8ffdf54d364bd91e4e74afd",
    )

    test_url, test_md5 = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test",
        "46815731e31c07f89422cf60de8738e7",
    )
