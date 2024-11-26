import functools
import glob
import multiprocessing
import os
import sys
from cgi import print_form
from dataclasses import dataclass
import time
from platform import system
from traceback import print_tb

import numpy as np
import pandas as pd
from pyarrow import feather as ft

from src.LinoSPAD2.functions import calc_diff as cd
from src.LinoSPAD2.functions import unpack as f_up
from src.LinoSPAD2.functions import utils
from src.LinoSPAD2.functions.calibrate import load_calibration_data


@dataclass
class DataParamsConfig:
    pixels: list
    path: str = ""
    daughterboard_number: str = ""
    motherboard_number: str = ""
    motherboard_number1: str = ""
    motherboard_number2: str = ""
    firmware_version: str = ""
    timestamps: int = 512
    delta_window: float = 50e3
    app_mask: bool = True
    include_offset: bool = True
    apply_calibration: bool = True
    absolute_timestamps: bool = False


def _calculate_timestamps_differences(files, data_params, path, pix_coor, pixels, calibration_matrix, offset_array):
    read_dat_files = [np.memmap(file, dtype=np.uint32) for file in files] # read all the files to avoid 1 by 1 reading

    for i in range(len(read_dat_files)):
        # unpack data from binary files
        data_all = f_up.unpack_binary_data(
            read_dat_files[i],
            calibration_matrix,
            offset_array,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )

        # calculate the differences and convert them to a pandas dataframe
        deltas_all = cd.calculate_differences_2212_fast(data_all, pixels, pix_coor)
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index").T

        # save the data to a feather file
        file_name = os.path.basename(files[i])
        output_file = os.path.join(path, str(file_name.replace('.dat', '.feather')))
        data_for_plot_df.reset_index(drop=True, inplace=True)
        ft.write_feather(data_for_plot_df, output_file)


def calculate_and_save_timestamp_differences_mp(
        num_of_files,
        files,
        output_directory,
        pixels,
        daughterboard_number,
        motherboard_number,
        firmware_version,
        timestamps,
        delta_window: float = 50e3,
        app_mask: bool = True,
        include_offset: bool = True,
        apply_calibration: bool = True,
        number_of_cores: int = 10,
):
    # check the firmware version and set the pixel coordinates accordingly
    if firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Generate a dataclass object
    data_params = DataParamsConfig(
        pixels=pixels,
        daughterboard_number=daughterboard_number,
        motherboard_number=motherboard_number,
        firmware_version=firmware_version,
        timestamps=timestamps,
        delta_window=delta_window,
        app_mask=app_mask,
        include_offset=include_offset,
        apply_calibration=apply_calibration,
        absolute_timestamps=True,
    )

    calibration_matrix, offset_array = None, None # initialize calibration matrix and offset array in case they are used

    # load calibration data if necessary
    if data_params.apply_calibration:
        path_calibration_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "params", "calibration_data", )
        calibration_matrix, offset_array = load_calibration_data(
            path_calibration_data,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            include_offset,
        )

    # Apply mask if necessary
    if data_params.app_mask:
        mask = utils.apply_mask(data_params.daughterboard_number, data_params.motherboard_number)
        if isinstance(data_params.pixels[0], int) and isinstance(data_params.pixels[1], int):
            pixels = [pix for pix in data_params.pixels if pix not in mask]
        else:
            pixels = [pix for pix in data_params.pixels[0] if pix not in mask]
            pixels.extend(pix for pix in data_params.pixels[1] if pix not in mask)

    print("Starting analysis of the files")
    start_time = time.time()

    processes = [] # list to store all the processes
    # Create processes (number of cores) and assign to each process its specified files chunk (files/number_of_cores)
    # each process will run the _calculate_timestamps_differences function with its own parameters
    # target: the function to be run
    # args: the arguments to be passed to the function
    for i in range(number_of_cores):
        p = multiprocessing.Process(
            target=_calculate_timestamps_differences,
            args=(files[i], data_params, output_directory, pix_coor, pixels, calibration_matrix,
                  offset_array))
        p.start()
        processes.append(p) # add the process to the list so we can wait for all of them to finish

    # wait for all processes to finish, and only then continue to the next step
    for process in processes:
        process.join()

    end_time = time.time()

    output_string = f"Parallel processing of {num_of_files} files (with each writing to its file) finished in: {round(end_time - start_time, 2)} s"
    print(output_string)
