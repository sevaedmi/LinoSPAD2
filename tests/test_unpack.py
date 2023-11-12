import unittest
import os

import numpy as np

from LinoSPAD2.functions.unpack import unpack_bin


class TestUnpackBin(unittest.TestCase):
    def test_valid_input(self):
        # Positive test case with valid inputs
        work_dir = r"{}".format(os.path.realpath(__file__) + "../../..")
        os.chdir(work_dir)
        file = r"tests/test_data/test_data_2212b.dat"
        db_num = "NL11"
        mb_num = "#33"
        timestamps = 300
        fw_ver = "2212b"
        inc_offset = False

        data_all = unpack_bin(
            file, db_num, mb_num, fw_ver, timestamps, inc_offset
        )

        # Assert the shape of the output data
        self.assertEqual(data_all.shape, (64, 300 * 300 + 300, 2))
        # Assert the data type of the output data
        self.assertEqual(data_all.dtype, np.longlong)


if __name__ == "__main__":
    unittest.main()
