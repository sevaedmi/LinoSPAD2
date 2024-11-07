import functools
import glob
import multiprocessing
import os
import sys
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


def _calculate_timestamps_differences(files, data_params, path, write_to_files, pix_coor, pixels, calibration_matrix, offset_array):
    for file in files:
        data_all = f_up.unpack_binary_data(
            file,
            calibration_matrix,
            offset_array,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )

        deltas_all = cd.calculate_differences_2212_fast(data_all, pixels, pix_coor)
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index").T

        if write_to_files:
            file_name = file.split('/')[-1]
            output_file = "/home/dmitrij/FJFI/LinoSPAD2/raw_data/delta_ts_data/" + str(file_name.replace('.dat', '.feather'))
            data_for_plot_df.reset_index(drop=True, inplace=True)
            ft.write_feather(data_for_plot_df, output_file)


def calculate_and_save_timestamp_differences_mp(
        num_of_files, files,output_directory,pixels,daughterboard_number,motherboard_number,firmware_version,timestamps,
        delta_window: float = 50e3,
        app_mask: bool = True,
        include_offset: bool = True,
        apply_calibration: bool = True,
        number_of_cores: int = 10,
        write_to_files: bool = True,
):
    if firmware_version == "2212s":pix_coor = np.arange(256).reshape(4, 64).T
    elif firmware_version == "2212b":pix_coor = np.arange(256).reshape(64, 4)
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

    print("Starting analysis of the files with writing to files set to:", write_to_files)
    start_time = time.time()

    # create processes (number of cores) and send them its assigned sublist of files
    processes = []
    for i in range(number_of_cores):
        p = multiprocessing.Process(
            target=_calculate_timestamps_differences,
            args=(files[i], data_params, output_directory, write_to_files, pix_coor, pixels, calibration_matrix, offset_array))
        processes.append(p)
        p.start()

    # wait for all processes to finish
    for process in processes:
        process.join()

    end_time = time.time()

    if write_to_files:
        output_string = f"Parallel processing of {num_of_files} files (with each writing to its file) finished in: {round(end_time - start_time, 2)} s"
    else:
        output_string = f"Parallel processing of {num_of_files} files (without writing any results) finished in: {round(end_time - start_time, 2)} s"
    print(output_string)
