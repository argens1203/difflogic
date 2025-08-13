import unittest
import pytest

from experiment.helpers.util import get_onehot_loader


class TestUtil(unittest.TestCase):

    def test_get_onehot_loader(self):
        input_dim = 8
        attribute_ranges = [2, 2, 2, 2]

        for x, _ in get_onehot_loader(input_dim, attribute_ranges, batch_size=2):
            print(x)

        self.assertEqual(1, 1)
