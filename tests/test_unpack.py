import unittest

import numpy as np

from LinoSPAD2.functions.unpack import unpack_bin


class TestUnpackBin(unittest.TestCase):
    def test_valid_input(self):
        # Positive test case with valid inputs
        file = r"tests/test_data/test_data_2212b.dat"
        board_number = "A5"
        timestamps = 200

        data_all = unpack_bin(file, board_number, timestamps)

        # Assert the shape of the output data
        self.assertEqual(data_all.shape, (64, 4020, 2))
        # Assert the data type of the output data
        self.assertEqual(data_all.dtype, np.longlong)


if __name__ == "__main__":
    unittest.main()
