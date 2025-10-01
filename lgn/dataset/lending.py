import numpy as np

from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer


class LendingDataset(CustomDataset, AutoTransformer):
    train_url, train_md5 = (
        "https://figshare.com/ndownloader/files/39316160",
        "87dfdf86ff1eebfb6a58f3ffcc45392c",
    )
    test_url, test_md5 = (
        "https://figshare.com/ndownloader/files/39495787",
        "32d4c6bcdda210da47e6aac2ea8ecae6",
    )

    converter = None
    label_encoder = None

    @classmethod
    def attributes(cls):
        return [
            # "issue_d",
            "sub_grade",
            "term",
            "home_ownership",
            "fico_range_low",
            "total_acc",
            "pub_rec",
            "revol_util",
            "annual_inc",
            # "int_rate",
            "dti",
            "purpose",
            "mort_acc",
            "loan_amnt",
            "application_type",
            # "installment",
            "verification_status",
            "pub_rec_bankruptcies",
            "addr_state",
            "initial_list_status",
            "fico_range_high",
            "revol_bal",
            # "id",
            "open_acc",
            "emp_length",
            # "loan_status",
            "time_to_earliest_cr_line",
        ]

    @classmethod
    def continuous_attributes(cls):
        return set(
            {
                "revol_util",
                "annual_inc",
                # "int_rate",
                "dti",
                "loan_amnt",
                # "installment",
                "revol_bal",
                "time_to_earliest_cr_line",
            }
        )

    @classmethod
    def discrete_attributes(cls):
        return set(cls.attributes()) - cls.continuous_attributes()

    @classmethod
    def bin_sizes(cls):
        return dict(
            {
                "revol_util": 5,
                "annual_inc": 5,
                "int_rate": 5,
                "dti": 5,
                "loan_amnt": 5,
                "installment": 5,
                "revol_bal": 5,
                "time_to_earliest_cr_line": 5,
            }
        )

    def __init__(
        self,
        root=None,
        split="train",
        transform=None,
    ):
        self.url = self.train_url if split == "train" else self.test_url
        self.md5 = self.train_md5 if split == "train" else self.test_md5

        super(LendingDataset, self).__init__(transform=transform, root=root)

    def load_data(self):
        raw_data = self.read_raw_data(
            self._get_fpath(),
            delimiter=",",
            select=lambda x: "?" not in x and "|" not in x,
        )

        # Cast as strings + Remove header
        features = raw_data[1:, :].astype(str)
        labels = raw_data[1:, -2].astype(str)  # Second last column (loan_status)

        # Remove issue_d, int_rate, installment, id, loan_status
        features = np.delete(features, [0, 9, 15, 22, 25], axis=1)

        self.raw_features = features.copy()

        self.features = LendingDataset.transform_feature(features)
        self.labels = LendingDataset.transform_label(labels)
        print(LendingDataset.get_attribute_ranges())
        input("Press Enter to continue...")
