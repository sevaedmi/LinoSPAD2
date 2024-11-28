import glob
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from src.LinoSPAD2.functions import calc_diff as cd
from src.LinoSPAD2.functions import utils
from src.LinoSPAD2.functions.calibrate import load_calibration_data
from pyarrow import feather as ft


@dataclass
class DataParamsConfig:
    pixels: list
    path: str = ""
    daughterboard_number: str = "B7d"
    motherboard_number: str = "#28"
    firmware_version: str = "2212s"
    timestamps: int = 1000
    delta_window: float = 50e3
    app_mask: bool = True
    include_offset: bool = False
    apply_calibration: bool = True
    absolute_timestamps: bool = False


def _unpack_binary_data(
        file,
        calibration_matrix,
        offset_array,
        timestamps: int = 512,
        include_offset: bool = False,
        apply_calibration: bool = True,
):
    # Unpack binary data

    # raw_data = np.memmap(file, dtype=np.uint32)
    # Timestamps are stored in the lower 28 bits
    data_timestamps = (file & 0xFFFFFFF).astype(np.int64)
    # Pixel address in the given TDC is 2 bits above timestamp
    data_pixels = ((file >> 28) & 0x3).astype(np.int8)
    # Check the top bit, assign '-1' to invalid timestamps
    data_timestamps[file < 0x80000000] = -1
    # Number of acquisition cycles in each data file
    cycles = len(data_timestamps) // (timestamps * 65)
    # Transform into a matrix of size 65 by cycles*timestamps
    data_pixels = (
        data_pixels.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, -1)
    )
    data_timestamps = (
        data_timestamps.reshape(cycles, 65, timestamps)
        .transpose((1, 0, 2))
        .reshape(65, -1)
    )

    # Cut the 65th TDC that does not hold any actual data from pixels
    data_pixels = data_pixels[:-1]
    data_timestamps = data_timestamps[:-1]
    insert_indices = np.linspace(
        timestamps, cycles * timestamps, cycles
    ).astype(np.int64)

    # Insert '-2' at the end of each cycle
    data_pixels = np.insert(
        data_pixels,
        insert_indices,
        -2,
        1,
    )
    data_timestamps = np.insert(
        data_timestamps,
        insert_indices,
        -2,
        1,
    )

    # Combine both matrices into a single one, where each cell holds pixel
    # coordinates in the TDC and the timestamp
    data_all = np.stack((data_pixels, data_timestamps), axis=2).astype(
        np.int64
    )
    if apply_calibration is False:
        data_all[:, :, 1] = data_all[:, :, 1] * 2500 / 140
    else:
        pix_coordinates = np.arange(256).reshape(64, 4)
        for i in range(256):
            # Transform pixel number to TDC number and pixel coordinates in
            # that TDC (from 0 to 3)
            tdc, pix = np.argwhere(pix_coordinates == i)[0]
            # Find data from that pixel
            ind = np.where(data_all[tdc].T[0] == pix)[0]
            # Cut non-valid timestamps ('-1's)
            ind = ind[data_all[tdc].T[1][ind] >= 0]
            if not np.any(ind):
                continue
        data_cut = data_all[tdc].T[1][ind]
        if (
                include_offset
        ):  # Apply calibration; offset is added due to how delta ts
            # are calculated
            data_all[tdc].T[1][ind] = (
                    (data_cut - data_cut % 140) * 2500 / 140
                    + calibration_matrix[i, (data_cut % 140)]
                    + offset_array[i]
            )
        else:
            data_all[tdc].T[1][ind] = (
                                              data_cut - data_cut % 140
                                      ) * 2500 / 140 + calibration_matrix[i, (data_cut % 140)]
    return data_all


def _calculate_timestamps_differences(
        files,
        data_params,
        path,
        pix_coor,
        pixels,
        calibration_matrix,
        offset_array,
):
    read_dat_files = [
        np.memmap(file, dtype=np.uint32) for file in files
    ]  # read all the files to avoid 1 by 1 reading

    for i, _ in enumerate(read_dat_files):
        # unpack data from binary files
        data_all = _unpack_binary_data(
            read_dat_files[i],
            calibration_matrix,
            offset_array,
            data_params.timestamps,
            data_params.include_offset,
            data_params.apply_calibration,
        )

        print("here")

        # calculate the differences and convert them to a pandas dataframe
        deltas_all = cd.calculate_differences_2212_fast(
            data_all, pixels, pix_coor
        )
        data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index").T

        # save the data to a feather file
        file_name = os.path.basename(files[i])

        output_file = os.path.join(
            path, str(file_name.replace(".dat", ".feather"))
        )
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
        include_offset: bool = False,
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

    calibration_matrix, offset_array = (
        None,
        None,
    )  # initialize calibration matrix and offset array in case they are used

    # load calibration data if necessary
    if data_params.apply_calibration:
        path_calibration_data = r"C:\\Users\\fintv\\Desktop\\FJFI\\LinoSPAD2\\src\\LinoSPAD2\\params\\calibration_data"
        calibration_matrix, offset_array = load_calibration_data(
            path_calibration_data,
            daughterboard_number,
            motherboard_number,
            firmware_version,
            include_offset,
        )

    # Apply mask if necessary
    if data_params.app_mask:
        mask = utils.apply_mask(
            data_params.daughterboard_number, data_params.motherboard_number
        )
        if isinstance(data_params.pixels[0], int) and isinstance(
                data_params.pixels[1], int
        ):
            pixels = [pix for pix in data_params.pixels if pix not in mask]
        else:
            pixels = [pix for pix in data_params.pixels[0] if pix not in mask]
            pixels.extend(
                pix for pix in data_params.pixels[1] if pix not in mask
            )

    print("Starting analysis of the files")
    start_time = time.time()

    processes = []  # list to store all the processes
    # Create processes (number of cores) and assign to each process
    # its specified files chunk (files/number_of_cores)
    # each process will run the _calculate_timestamps_differences
    # function with its own parameters target: the function to be run
    # args: the arguments to be passed to the function
    for i in range(number_of_cores):
        p = multiprocessing.Process(
            target=_calculate_timestamps_differences,
            args=(
                files[i],
                data_params,
                output_directory,
                pix_coor,
                pixels,
                calibration_matrix,
                offset_array,
            ),
        )
        p.start()
        processes.append(
            p
        )  # add the process to the list so we can wait for all of
        # them to finish

    # wait for all processes to finish, and only then continue to the
    # next step
    for process in processes:
        process.join()

    end_time = time.time()

    output_string = f"Parallel processing of {num_of_files} files (with each writing to its file) finished in: {round(end_time - start_time, 2)} s"
    print(output_string)


def parallel(path: str, num_of_cores):
    # Find all .dat files in the specified path
    files = glob.glob(os.path.join(path, "*.dat*"))

    num_of_files = len(files)
    # Divide the files into sublists for each process, number of
    # sublists = number of cores
    files = np.array_split(files, num_of_cores)

    output_directory = os.path.join(path, "delta_ts_data")

    calculate_and_save_timestamp_differences_mp(
        num_of_files,
        files,
        output_directory,
        pixels=[139, 167],
        daughterboard_number="B7d",
        motherboard_number="#28",
        firmware_version="2212s",
        timestamps=1000,
        include_offset=False,
        number_of_cores=num_of_cores,
    )


def _merge_files(path: str):
    path = os.path.join(path, "delta_ts_data")

    # Find all .feather files in the directory
    feather_files = [
        path + "/" + f for f in os.listdir(path) if f.endswith(".feather")
    ]

    # Start the timer
    start = time.time()

    # Use pandas to concatenate all feather files into a single dataframe
    dfs = [pd.read_feather(file) for file in feather_files]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged dataframe back to a single feather file with
    # current date and time
    cur_date_and_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_file_path = path + "/" + cur_date_and_time + "_merged.feather"
    merged_df.to_feather(output_file_path)

    # End the timer
    finish = time.time()

    # Print the time taken
    print(
        f"Sequential merging of {len(feather_files)} files "
        f"finished in: {round(finish - start, 2)} s"
    )


def _delete_results(path: str):
    try:
        path = os.path.join(path, "delta_ts_data")
        # Find all .feather files in the directory
        feather_files = [
            path + "/" + f for f in os.listdir(path) if f.endswith(".feather")
        ]

        # Delete all the feather files
        for file in feather_files:
            os.remove(file)

        print("Deleted all the feather files")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    # Get the current script directory
    current_directory = Path(__file__).parent
    # Define the path to the 'raw_data' directory
    path = str(current_directory / 'isolated_data')

    _delete_results(path)
    parallel(path, 10)
    _merge_files(path)
