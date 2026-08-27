import random
import unittest

import numpy as np

from src.utils.reproducibility import set_global_seed


class TestReproducibility(unittest.TestCase):
    def test_global_seed_repeats_python_and_numpy_sequences(self):
        set_global_seed(42)
        first = (random.random(), np.random.random())
        set_global_seed(42)
        second = (random.random(), np.random.random())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
