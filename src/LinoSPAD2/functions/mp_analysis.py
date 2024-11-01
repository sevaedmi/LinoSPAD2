import functools
import glob
import multiprocessing
import os
import sys
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from pyarrow import feather as ft, output_stream

from src.LinoSPAD2.functions import calc_diff as cd
from src.LinoSPAD2.functions import unpack as f_up
from src.LinoSPAD2.functions import utils


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


def _calculate_timestamps_differences(file: str, data_params: DataParamsConfig, path: str, write_to_files: bool):
    # Process the pixel coordinates based on firmware version
    if data_params.firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif data_params.firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    # Apply mask if necessary
    if data_params.app_mask:
        mask = utils.apply_mask(
            data_params.daughterboard_number, data_params.motherboard_number
        )
        if isinstance(data_params.pixels[0], int) and isinstance(data_params.pixels[1], int):
            pixels = [pix for pix in data_params.pixels if pix not in mask]
        else:
            pixels = [pix for pix in data_params.pixels[0] if pix not in mask]
            pixels.extend(pix for pix in data_params.pixels[1] if pix not in mask)

    # Unpack the binary data from the file
    data_all = f_up.unpack_binary_data(
        file,
        data_params.daughterboard_number,
        data_params.motherboard_number,
        data_params.firmware_version,
        data_params.timestamps,
        data_params.include_offset,
        data_params.apply_calibration,
    )

    # Calculate the timestamp differences
    deltas_all = cd.calculate_differences_2212_fast(data_all, pixels, pix_coor)

    # Convert the result to a DataFrame
    data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index").T

    if write_to_files:
        # Create a unique output file name for each process based on the input file name
        output_file = file.replace('.dat', '.feather')
        output_file = os.path.join(path, output_file)  # Adjust path if needed

        # Write the results directly to a unique Feather file
        data_for_plot_df.reset_index(drop=True, inplace=True)
        ft.write_feather(data_for_plot_df, output_file)


def calculate_and_save_timestamp_differences_mp(
        path: str,
        pixels: list,
        rewrite: bool,
        daughterboard_number: str,
        motherboard_number: str,
        firmware_version: str,
        timestamps: int,
        delta_window: float = 50e3,
        app_mask: bool = True,
        include_offset: bool = True,
        apply_calibration: bool = True,
        chunksize: int = 1,
        number_of_cores: int = 1,
        maxtasksperchild: int = None,
        write_to_files: bool = True,
):
    # Parameter type checks
    if not isinstance(pixels, list): raise TypeError("'pixels' should be a list of integers or a list of two lists")
    if not isinstance(firmware_version, str): raise TypeError(
        "'firmware_version' should be string, '2212s', '2212b' or '2208'")
    if not isinstance(rewrite, bool): raise TypeError("'rewrite' should be boolean")
    if not isinstance(daughterboard_number, str): raise TypeError("'daughterboard_number' should be string")

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

    os.chdir(path)

    # Find all LinoSPAD2 data files
    files = glob.glob("*.dat*")
    output_directory = os.path.join(path, "delta_ts_data")

    print("Starting analysis of the files with writing to files set to:", write_to_files)

    start_time = time.time()

    # Create a pool of processes, each one will write to its own file if write_to_files is True
    with multiprocessing.Pool(processes=number_of_cores, maxtasksperchild=maxtasksperchild) as pool:
        # Create a partial function with fixed arguments for process_file
        partial_process_file = functools.partial(_calculate_timestamps_differences, path=output_directory,
                                                 data_params=data_params, write_to_files=write_to_files)

        # Start the multicore analysis of the files
        pool.map(partial_process_file, files, chunksize=chunksize)

    end_time = time.time()

    if write_to_files:
        output_string = f"Parallel processing of {len(files)} files (with each writing to its file) finished in: {round(end_time - start_time, 2)} s"
    else:
        output_string = f"Parallel processing of {len(files)} files (without writing any results) finished in: {round(end_time - start_time, 2)} s"
    print(output_string)
