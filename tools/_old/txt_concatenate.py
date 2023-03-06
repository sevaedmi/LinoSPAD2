"""Combine multiple txt files with data into a single one."""

# Note: does not work with ~14 files -> large memory consumption

import glob
import os

PATH = (
    "C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"
    "40 ns window, 20 MHz clock, 10 cycles/10 lines of data"
)
os.chdir(PATH)

filenames = glob.glob("*txt*")


with open("all_data.txt", "w") as outfile:
    for filename in filenames:
        with open(filename) as infile:
            contents = infile.read()
            outfile.write(contents)
