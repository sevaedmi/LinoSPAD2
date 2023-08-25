import unittest
import numpy as np
import os
import shutil

from LinoSPAD2.functions.delta_t import deltas_save, delta_cp
from LinoSPAD2.functions.fits import fit_wg


class TestDeltasFull(unittest.TestCase):
    def setUp(self):
        # Set up test variables
        self.path = "tests/test_data"
        self.pixels = np.arange(0, 5, 1)
        self.board_number = "A5"
        self.fw_ver = "2212b"
        self.timestamps = 200
        self.delta_window = 10e3
        self.rewrite = True
        self.range_left = -10e3
        self.range_right = 10e3
        self.same_y = False

    def test_a_deltas_save_positive(self):
        # Test positive case for deltas_save function
        work_dir = r"{}".format(os.path.realpath(__file__) + "../../..")
        os.chdir(work_dir)
        deltas_save(
            self.path,
            self.pixels,
            self.rewrite,
            self.board_number,
            self.fw_ver,
            self.timestamps,
            self.delta_window,
        )

        # Check if the csv file is created
        self.assertTrue(
            os.path.isfile("delta_ts_data/test_data_2212b-test_data_2212b.csv")
        )

    # Negative test case
    # Invalid firmware version
    def test_b_deltas_save_negative(self):
        work_dir = r"{}".format(os.path.realpath(__file__) + "../../..")
        os.chdir(work_dir)
        # Test negative case for deltas_save function
        with self.assertRaises(TypeError):
            deltas_save(
                self.path,
                self.pixels,
                "2212",
                self.board_number,
                self.fw_ver,
                self.timestamps,
                self.delta_window,
            )

    def test_c_delta_cp(self):
        # Test case for delta_cp function
        # Positive test case
        os.chdir(r"{}".format(os.path.realpath(__file__) + "/../.."))
        delta_cp(
            self.path,
            self.pixels,
            self.rewrite,
            self.range_left,
            self.range_right,
            self.same_y,
        )

        # Check if the plot file is created
        self.assertTrue(
            os.path.isfile(
                "results/delta_t/test_data_2212b-test_data_2212b_delta_t_grid.png"
            )
        )

    # TODO: need larger data set to get more delta ts
    # def test_d_fit_wg_positive(self):
    #     # Test with valid input
    #     os.chdir(r"{}".format(os.path.realpath(__file__) + "/../.."))
    #     pix_pair = [2, 4]
    #     window = 5e3
    #     step = 1

    #     # Call the function
    #     fit_wg(self.path, pix_pair, window, step)

    #     # Assert that the function runs without raising any exceptions
    #     self.assertTrue(
    #         os.path.isfile(
    #             "results/fits/test_data_2212b-test_data_2212b_pixels_2,4_fit.png"
    #         )
    #     )

    def tearDownClass():
        # Clean up after tests
        os.chdir(r"{}".format(os.path.realpath(__file__) + "/.."))
        shutil.rmtree("test_data/delta_ts_data")
        shutil.rmtree("test_data/results")


if __name__ == "__main__":
    unittest.main()
