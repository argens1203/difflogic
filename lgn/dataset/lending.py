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
                # "issue_d",
                # "sub_grade",
                # "term",
                # "home_ownership",
                "fico_range_low",
                "total_acc",
                "pub_rec",
                "revol_util",
                "annual_inc",
                # "int_rate",
                "dti",
                # "purpose",
                "mort_acc",
                "loan_amnt",
                # "application_type",
                # "installment",
                # "verification_status",
                "pub_rec_bankruptcies",
                # "addr_state",
                # "initial_list_status",
                "fico_range_high",
                "revol_bal",
                # "id",
                "open_acc",
                "emp_length",
                # "loan_status",
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
                # "issue_d": 5,
                # "sub_grade": 5,
                # "term": 5,
                # "home_ownership": 5,
                "fico_range_low": 5,
                "total_acc": 5,
                "pub_rec": 5,
                "revol_util": 5,
                "annual_inc": 5,
                # "int_rate": 5,
                "dti": 5,
                # "purpose": 5,
                "mort_acc": 5,
                "loan_amnt": 5,
                # "application_type": 5,
                # "installment": 5,
                # "verification_status": 5,
                "pub_rec_bankruptcies": 5,
                # "addr_state": 5,
                # "initial_list_status": 5,
                "fico_range_high": 5,
                "revol_bal": 5,
                # "id": 5,
                "open_acc": 5,
                "emp_length": 5,
                # "loan_status": 5,
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
        self.split = split

        super(LendingDataset, self).__init__(transform=transform, root=root)

    def filter_row(self, features, labels):
        if self.split == "train":
            return features, labels

        # Create mask for rows without missing data
        valid_rows_mask = np.ones(len(features), dtype=bool)

        # Check features for missing data
        for i, row in enumerate(features):
            for cell in row:
                if str(cell).strip().upper() in [
                    "ANY",
                    "NONE",
                    "",
                ]:  # ANY is not seen at all in training set
                    valid_rows_mask[i] = False
                    break

        # Also check labels for missing data
        for i, label in enumerate(labels):
            if (
                str(label).strip().upper()
                in [
                    "ANY",
                    "NONE",
                    "",
                ]
                or str(label).strip() == ""
            ):
                valid_rows_mask[i] = False

        # Filter both features and labels using the mask
        features = features[valid_rows_mask]
        labels = labels[valid_rows_mask]
        return features, labels

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

        features, labels = self.filter_row(features, labels)

        self.raw_features = features.copy()

        self.features = LendingDataset.transform_feature(features)
        self.labels = LendingDataset.transform_label(labels)
