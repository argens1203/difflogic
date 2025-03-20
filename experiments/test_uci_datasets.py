import unittest
import numpy as np
from uci_datasets import IrisDataset


class TestIrisDataset(unittest.TestCase):
    def test_convert_sample_to_feature_vector(self):
        sample = ["5.1", "3.5", "1.4", "0.2", "Iris-setosa"]
        maxes = np.array([7.9, 4.4, 6.9, 2.5], dtype=float)
        mins = np.array([4.3, 2.0, 1.0, 0.1], dtype=float)

        vec = IrisDataset.convert_sample_to_feature_vector(
            sample, maxes=maxes, mins=mins
        )
        expected = [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        for a, b in zip(vec, expected):
            self.assertEqual(a, b)

    def test_convert_sample_to_feature_vector_min_max(self):
        sample = ["5.1", "3.5", "1.4", "0.2", "Iris-setosa"]
        maxes = np.array([5.1, 4.4, 6.9, 2.5], dtype=float)
        mins = np.array([4.3, 2.0, 1.4, 0.1], dtype=float)

        vec = IrisDataset.convert_sample_to_feature_vector(
            sample, maxes=maxes, mins=mins
        )
        expected = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        i = 0
        for a, b in zip(vec, expected):
            self.assertEqual(a, b)
            i = i + 1
