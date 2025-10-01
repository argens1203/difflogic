from typing import Literal, Union
import numpy as np

from .custom_dataset import CustomDataset
from .auto_transformer import AutoTransformer

ReciType = Union[Literal["recid"], Literal["violent_recid"]]


class CompasDataset(CustomDataset, AutoTransformer):
    url, md5 = (
        "https://github.com/propublica/compas-analysis/blob/master/compas-scores.csv?raw=true",
        "14c38b686ad790afcb8ef2ab4d585ed7",
    )

    converter = None
    label_encoder = None

    @classmethod
    def attributes(cls):
        return [
            # "id",
            # "name",
            # "first",
            # "last",
            # "compas_screening_date",
            "sex",
            # "dob",
            # "age",
            "age_cat",
            "race",
            "juv_fel_count",
            # "decile_score",
            "juv_misd_count",
            "juv_other_count",
            "priors_count",
            # "days_b_screening_arrest",
            # "c_jail_in",
            # "c_jail_out",
            # "c_case_number",
            # "c_offense_date",
            # "c_arrest_date",
            # "c_days_from_compas",
            "c_charge_degree",
            # "c_charge_desc",
            # "is_recid", <--- OUTPUT
            # "num_r_cases", <--- EMPTY
            # "r_case_number",
            # "r_charge_degree",
            # "r_days_from_arrest",
            # "r_offense_date",
            # "r_charge_desc",
            # "r_jail_in",
            # "r_jail_out",
            # "is_violent_recid", <--- OUTPUT
            # "num_vr_cases", <--- EMPTY
            # "vr_case_number",
            # "vr_charge_degree",
            # "vr_offense_date",
            # "vr_charge_desc",
            # "v_type_of_assessment",
            # "v_decile_score",
            # "v_score_text",
            # "v_screening_date",
            # "type_of_assessment",
            # "decile_score",
            # "score_text",
            # "screening_date",
        ]

    @classmethod
    def continuous_attributes(cls):
        return set(
            {
                "juv_fel_count",
                "juv_misd_count",
                "juv_other_count",
                "priors_count",
                # "days_b_screening_arrest",
                # "num_r_cases",
                # "num_vr_cases",
            }
        )

    @classmethod
    def discrete_attributes(cls):
        return set(cls.attributes()) - cls.continuous_attributes()

    @classmethod
    def bin_sizes(cls):
        return dict(
            {
                "juv_fel_count": 5,
                "juv_misd_count": 5,
                "juv_other_count": 5,
                "priors_count": 5,
                # "days_b_screening_arrest": 5,
                # "num_r_cases": 5,
                # "num_vr_cases": 5,
            }
        )

    def __init__(
        self,
        root=None,
        split="train",
        output: ReciType = "recid",
        transform=None,
    ):
        # self.url = self.train_url if split == "train" else self.test_url
        # self.md5 = self.train_md5 if split == "train" else self.test_md5
        self.output = output
        super(CompasDataset, self).__init__(transform=transform, root=root)

    def load_data(self):
        raw_data = self.read_raw_data(
            self._get_fpath(),
            delimiter=",",
            select=lambda x: "?" not in x and "|" not in x and len(x.strip()) > 0,
        )

        # # Remove final empty line
        # raw_data = raw_data[:-1, :]

        # Cast as strings + Remove header
        features = raw_data[1:, :].astype(str)
        labels = (
            raw_data[1:, 24].astype(str)  # Recidivism
            if self.output == "recid"
            else raw_data[1:, 33].astype(str)  # Violent Recidivism
        )

        # Remove unused attributes
        cols_to_del = np.r_[0:5, 6:8, 11, 15:22, 23:47]
        features = np.delete(features, cols_to_del, axis=1)

        # # Create mask for rows without missing data
        # valid_rows_mask = np.ones(len(features), dtype=bool)

        # # Check features for missing data
        # for i, row in enumerate(features):
        #     for cell in row:
        #         if str(cell).strip().lower() == "":
        #             print(row, cell)
        #             valid_rows_mask[i] = False
        #             break

        # # Filter both features and labels using the mask
        # features = features[valid_rows_mask]
        # labels = labels[valid_rows_mask]

        self.raw_features = features.copy()

        self.features = CompasDataset.transform_feature(features)
        self.labels = CompasDataset.transform_label(labels)
