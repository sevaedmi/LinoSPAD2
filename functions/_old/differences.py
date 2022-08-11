"""Calculate differences in timestamps between all pixels in each acquistion
cycle.

Works with both 'txt' and 'dat' files.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This script requires that `pandas` be installed within the Python
environment you are running this script in.

The output is saved in the `results` directory, in the case there is no such
folder, it is created where the data are stored.

This file can also be imported as a module and contains the following
functions:

    * timestamp_diff - calculates the differences in timestamps between all
    pixels
    * timestamp_diff_flex - calculates the differences in timestamps between
    all pixels; requires manual input of number of data points in a single
    acquisition cycle

"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import functions.unpack as f_up


def timestamp_diff(path):
    """Calculates the differences in timestamps between all pixels in each
    acquistion cycle. Data are saved into a csv file.

    Parameters
    ----------
    path : str
        Location of data files from LinoSPAD2.
    Returns
    -------
    None.
    """

    os.chdir(path)

    if "binary" in path:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.dat*')
        for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            data_matrix = f_up.unpack_binary_10(DATA_FILES[r])
            # dimensions for matrix of timestamp differences
            minuend = len(data_matrix) - 1  # i=255
            lines_of_data = len(data_matrix[0])  # j=10*11999 (lines of data
            # * number of acq cycles)
            subtrahend = len(data_matrix) - 2  # k=254
            timestamps = 10  # lines of data in the acq cycle

            output = []

            for i in tqdm(range(minuend)):
                acq = 0  # number of acq cycle
                for j in range(lines_of_data):
                    if data_matrix[i][j] == -1:
                        continue
                    if j % 10 == 0:
                        acq = acq + 1  # next acq cycle
                    for k in range(subtrahend):
                        if k <= i:
                            continue  # to avoid repetition: 2-1, 153-45 etc.
                        for p in range(timestamps):
                            n = 10*(acq-1) + p
                            if data_matrix[k][n] == -1:
                                continue
                            # collecting timestamp differences within 10ns
                            # range
                            elif np.abs(data_matrix[i][j]
                                        - data_matrix[k][n]) > 10000:
                                continue
                            else:
                                output.append(data_matrix[i][j]
                                              - data_matrix[k][n])

            # open csv file with results and add a column
            print("\nSaving the data to the 'results' folder.")
            output_csv = pd.Series(output)
            os.chdir('results')
            filename = glob.glob('*timestamp_diff*')
            if not filename:
                with open('timestamp_diff.csv', 'w'):
                    filename = glob.glob('*timestamp_diff*')
                    pass

            try:
                csv = pd.read_csv(filename[0])
                csv.insert(loc=0, column="{}".format(DATA_FILES[r]),
                           value=output_csv, allow_duplicates=True)
                csv.to_csv(filename[0], index=False)
            except Exception:
                output_csv.to_csv(filename[0], header=[
                    '{}'.format(DATA_FILES[r])], index=False)
                # output_csv.to_csv(filename[0], index=False)
            os.chdir('..')

    else:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.txt*')
        for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            data_matrix = f_up.unpack_txt_10(DATA_FILES[r])
            # dimensions for matrix of timestamp differences
            minuend = len(data_matrix) - 1  # i=255
            lines_of_data = len(data_matrix[0])  # j=10*11999 (lines of data
            # * number of acq cycles)
            subtrahend = len(data_matrix) - 2  # k=254
            timestamps = 10

            output = []

            for i in range(minuend):
                acq = 0  # number of acq cycle
                for j in range(lines_of_data):
                    if j % 10 == 0:
                        acq = acq + 1  # next acq cycle
                    for k in range(subtrahend):
                        if k <= i:
                            continue  # to avoid repetition: 2-1, 153-45 etc.
                        for p in range(timestamps):
                            n = 10*(acq-1) + p
                            if data_matrix[i][j] == -1 or \
                               data_matrix[k][n] == -1:
                                continue
                            else:
                                output.append(data_matrix[i][j]
                                              - data_matrix[k][n])

            # open csv file with results and add a column
            print("\nSaving the data to the 'results' folder.")
            output_csv = pd.Series(output)
            os.chdir('results')
            filename = glob.glob('*timestamp_diff*')
            if not filename:
                with open('timestamp_diff.csv', 'w'):
                    filename = glob.glob('*timestamp_diff*')
                    pass

            try:
                csv = pd.read_csv(filename[0])
                csv.insert(loc=0, column="{}".format(DATA_FILES[r]),
                           value=output_csv, allow_duplicates=True)
                csv.to_csv(filename[0], index=False)
            except Exception:
                output_csv.to_csv(filename[0], header=[
                    '{}'.format(DATA_FILES[r])], index=False)
            os.chdir('..')


def timestamp_diff_flex(path, lod):
    """Calculates the differences in timestamps between all pixels in each
    acquistion cycle. Data are saved into a csv file.

    Parameters
    ----------
    path : str
        Location of data files from LinoSPAD2.
    lod : int
        Lines of data per acquisition cycle in the 'dat' file.
    Returns
    -------
    None.
    """

    os.chdir(path)

    if "binary" in path:
        # find all data files
        DATA_FILES = glob.glob('*acq*'+'*.dat*')
        for r in tqdm(range(len(DATA_FILES)), desc='Calculating'):
            data_matrix = f_up.unpack_binary_flex(DATA_FILES[r], lod)
            # dimensions for matrix of timestamp differences
            minuend = len(data_matrix) - 1  # i=255
            lines_of_data = len(data_matrix[0])  # j=10*11999 (lines of data
            # * number of acq cycles)
            subtrahend = len(data_matrix) - 2  # k=254
            timestamps = lod  # lines of data in the acq cycle

            output = []

            for i in tqdm(range(minuend)):
                acq = 0  # number of acq cycle
                for j in range(lines_of_data):
                    if data_matrix[i][j] == -1:
                        continue
                    if j % lod == 0:
                        acq = acq + 1  # next acq cycle
                    for k in range(subtrahend):
                        if k <= i:
                            continue  # to avoid repetition: 2-1, 153-45 etc.
                        for p in range(timestamps):
                            n = lod*(acq-1) + p
                            if data_matrix[k][n] == -1:
                                continue
                            elif np.abs(data_matrix[i][j]
                                        - data_matrix[k][n]) > 10000:
                                continue
                            else:
                                output.append(data_matrix[i][j]
                                              - data_matrix[k][n])

            # open csv file with results and add a column
            print("\nSaving the data to the 'results' folder.")
            output_csv = pd.Series(output)
            os.chdir('results')
            filename = glob.glob('*timestamp_diff*')
            if not filename:
                with open('timestamp_diff.csv', 'w'):
                    filename = glob.glob('*timestamp_diff*')
                    pass

            try:
                csv = pd.read_csv(filename[0])
                csv.insert(loc=0, column="{}".format(DATA_FILES[r]),
                           value=output_csv, allow_duplicates=True)
                csv.to_csv(filename[0], index=False)
            except Exception:
                output_csv.to_csv(filename[0], header=[
                    '{}'.format(DATA_FILES[r])], index=False)
                # output_csv.to_csv(filename[0], index=False)
            os.chdir('..')
