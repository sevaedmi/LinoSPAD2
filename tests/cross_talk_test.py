import glob
import os
import shutil
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(r"")
from functions.cross_talk import collect_ct, plot_ct


class TestCollectCT(unittest.TestCase):
    def setUp(self):
        # Set up test variables
        self.path = "tests/test_data"
        self.pixels = np.arange(0, 20, 1)
        self.board_number = "A5"
        self.timestamps = 200
        self.delta_window = 10e3

    def test_collect_ct_positive(self):
        # Test positive case of collect_ct function
        collect_ct(
            self.path,
            self.pixels,
            self.board_number,
            self.timestamps,
            self.delta_window,
        )
        # Check if the output file is created and has the correct number of rows
        file = glob.glob("*CT_data_*.csv*")[0]
        data = pd.read_csv(file, header=None)
        self.assertEqual(len(data), 20)

    def test_collect_ct_negative(self):
        # Test negative case of collect_ct function
        with self.assertRaises(TypeError):
            collect_ct(
                self.path, self.pixels, 123, self.timestamps, self.delta_window
            )


class TestPlotCT(unittest.TestCase):
    def setUp(self):
        # Set up test variables
        self.path = "tests/test_data"
        self.pix1 = 0
        self.scale = "linear"

    def test_plot_ct_positive(self):
        # Test positive case of plot_ct function
        work_dir = r"{}".format(os.path.realpath(__file__) + "../../..")
        os.chdir(work_dir)
        plot_ct(self.path, self.pix1, self.scale)
        # Check if the plot file is created
        plot_name = "test_data_2212b.dat_test_data_2212b.dat"
        plot_file = "{plot}_{pix}.png".format(plot=plot_name, pix=self.pix1)
        self.assertTrue(os.path.exists(plot_file))

    def test_plot_ct_negative(self):
        # Test negative case of plot_ct function
        with self.assertRaises(FileNotFoundError):
            plot_ct("nonexistent_folder", self.pix1, self.scale)

    def tearDownClass():
        # Clean up after tests
        os.chdir(r"{}".format(os.path.realpath(__file__) + "/.."))
        shutil.rmtree("test_data/cross_talk_data")
        shutil.rmtree("test_data/results")


if __name__ == "__main__":
    unittest.main()
