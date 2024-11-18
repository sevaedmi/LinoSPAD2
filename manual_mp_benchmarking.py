import glob
import os
import time
from os import cpu_count

import numpy as np
import pandas as pd
from src.LinoSPAD2.functions import delta_t, mp_analysis, manual_mp_analysis
from pathlib import Path

import cProfile
import pstats
import io


def sequential():
    # Get the current script directory
    current_directory = Path(__file__).parent
    # Define the path to the 'raw_data' directory
    path = str(current_directory / 'tmp_raw_data')
    start = time.time()
    delta_t.calculate_and_save_timestamp_differences_fast(
        path,
        pixels=[144, 171],
        rewrite=True,
        daughterboard_number="NL11",
        motherboard_number="#33",
        firmware_version="2212b",
        timestamps=300,
        include_offset=False,
    )
    finish = time.time()
    print(f"{finish - start} s")


def parallel(writing_to_files, num_of_cores):
    # Get the current script directory
    current_directory = Path(__file__).parent
    # Define the path to the 'raw_data' directory
    path = str(current_directory / 'tmp_raw_data')

    # Find all .dat files in the specified path
    files = glob.glob(os.path.join(path, "*.dat*"))
    num_of_files = len(files)
    # Divide the files into sublists for each process, number of sublists = number of cores
    files = np.array_split(files, num_of_cores)

    output_directory = os.path.join(path, "delta_ts_data")

    manual_mp_analysis.calculate_and_save_timestamp_differences_mp(
        num_of_files,
        files,
        output_directory,
        pixels=[144, 171],
        daughterboard_number="NL11",
        motherboard_number="#33",
        firmware_version="2212b",
        timestamps=300,
        include_offset=False,
        number_of_cores=num_of_cores,
        write_to_files=writing_to_files,
    )


def merge_files():
    # Get the current script directory
    current_directory = Path(__file__).parent
    # Define the path to the 'raw_data' directory
    path = str(current_directory / 'tmp_raw_data' / 'delta_ts_data')

    # Find all .feather files in the directory
    feather_files = [path + '/' + f for f in os.listdir(path) if f.endswith('.feather')]

    # Start the timer
    start = time.time()

    # Use pandas to concatenate all feather files into a single dataframe
    dfs = [pd.read_feather(file) for file in feather_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged dataframe back to a single feather file
    output_file_path = path + '/merged.feather'
    merged_df.to_feather(output_file_path)

    # End the timer
    finish = time.time()

    # Print the time taken
    print(f"Sequential merging of {len(feather_files)} files finished in: {round(finish - start, 2)} s")


def delete_results():
    # Get the current script directory
    current_directory = Path(__file__).parent
    # Define the path to the 'raw_data' directory
    path = str(current_directory / 'tmp_raw_data' / 'delta_ts_data')

    # Find all .feather files in the directory
    feather_files = [path + '/' + f for f in os.listdir(path) if f.endswith('.feather')]

    # Delete all the feather files
    for file in feather_files:
        os.remove(file)

    print("Deleted all the feather files")


if __name__ == "__main__":
    #delete_results()
    #sequential()
    #merge_files()
    #delete_results()

    delete_results()
    parallel(True, 10)

    merge_files()
    delete_results()

    # Manually done with 7 cores 800 files = 630 s
    # 800 files in 960 s, sequential
