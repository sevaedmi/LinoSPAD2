"""Combine multiple txt files with data into a single one."""

import glob
import os

os.chdir("C:/Users/bruce/Documents/Quantum astrometry/LinoSPAD/Software/Data/"
         "10 ns window, 50 MHz clock, 10 cycles")

filenames = glob.glob('*txt*')


with open("output_file.txt", "w") as outfile:
    for filename in filenames:
        with open(filename) as infile:
            contents = infile.read()
            outfile.write(contents)
