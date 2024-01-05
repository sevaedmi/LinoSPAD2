"""This module contains functions for LS2 data analysis that utilize
the multiprocessing Python library for speeding up the analysis
by using all available CPU cores instead of a single one.

This module can be imported with the following functions:

* calculate_and_save_timestamp_differences_mp - unpacks the binary data,
    calculates timestamp differences, and saves into a '.feather' file.
    Works with firmware versions '2208' and '2212b'. The multiprocessing
    version that utilizes all available CPU cores.

"""

import functools
import glob
import multiprocessing
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyarrow import feather as ft

from LinoSPAD2.functions import calc_diff as cd
from LinoSPAD2.functions import unpack as f_up
from LinoSPAD2.functions import utils

# from tqdm import tqdm


@dataclass
class DataParamsConfig:
    """Configuration parameters for timestamp differences calculation.

    Parameters
    ----------
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version : str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    app_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while apply_offset_calibration is set to 'False', only the TDC
        calibration is applied. The default is True.
    """

    pixels: list
    daughterboard_number: str
    motherboard_number: str
    firmware_version: str
    timestamps: int = 512
    delta_window: float = 50e3
    app_mask: bool = True
    include_offset: bool = True
    apply_calibration: bool = True


def calculate_timestamps_differences(
    file: str,
    result_queue: multiprocessing.Queue,
    data_params: DataParamsConfig,
) -> None:
    """Unpack data and collect timestamps differences for single file.

    Unpack the '.dat' data file, collect timestamps differences in the
    given window ("delta_window") and add them to the shared
    interprocess queue for further manipulations.

    Parameters
    ----------
    file : str
        Data file name, relative path.
    result_queue : multiprocessing.Queue
        Interprocess queue created using the multiprocessing.Queue method.
    data_params : DataParamsConfig
        An instance of the DataParamsConfig dataclass containing various
        configuration parameters for data processing.

    Returns
    -------
    None.
    """

    if data_params.firmware_version == "2212s":
        pix_coor = np.arange(256).reshape(4, 64).T
    elif data_params.firmware_version == "2212b":
        pix_coor = np.arange(256).reshape(64, 4)
    else:
        print("\nFirmware version is not recognized.")
        sys.exit()

    if data_params.app_mask is True:
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

    deltas_all = {}

    data_all = f_up.unpack_binary_data(
        file,
        data_params.daughterboard_number,
        data_params.motherboard_number,
        data_params.firmware_version,
        data_params.timestamps,
        data_params.include_offset,
        data_params.apply_calibration,
    )

    deltas_all = cd.calculate_differences_2212(data_all, pixels, pix_coor)

    data_for_plot_df = pd.DataFrame.from_dict(deltas_all, orient="index")

    result_queue.put(data_for_plot_df.T)


def write_results_to_feather(result_queue, feather_file, lock) -> None:
    """Save or append data to a Feather file.

    Parameters
    ----------
    result_queue : multiprocessing.Queue
        Interprocess queue containing data to be saved or appended to
        the Feather file.
    feather_file : str
        Absolute path to the Feather file.
    lock : multiprocessing.Lock
        Shared interprocess lock for synchronizing access to the Feather
        file.

    Notes
    -----
    The data in `result_queue` is retrieved and combined with the
    existing data in the Feather file, or a new file is created if it
    doesn't exist.
    The process is synchronized using the shared `lock` to prevent
    conflicts during file access.
    """
    while True:
        result_df = result_queue.get()
        if result_df is None:
            break

        # Use a lock to prevent conflicts when writing to the file
        with lock:
            if os.path.exists(feather_file):
                existing_data = ft.read_feather(feather_file)
                combined_data = pd.concat([existing_data, result_df], axis=0)
            else:
                combined_data = result_df.copy()

            # Reset the index to avoid issues during feather.write_feather
            combined_data.reset_index(drop=True, inplace=True)

            # Write the combined data to the Feather file
            ft.write_feather(combined_data, feather_file)


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
    chunksize: int = 20,
) -> None:
    """Unpack data and collect timestamps differences using all CPU cores.

    Unpack data files and collect timestamps differences using all
    available CPU cores to speed up the process, while saving the results
    to a single Feather file on the go.

    Parameters
    ----------
    path : str
        Path to data files.
    pixels : list
        List of pixel numbers for which the timestamp differences should
        be calculated and saved or list of two lists with pixel numbers
        for peak vs. peak calculations.
    rewrite : bool
        Switch for rewriting the '.feather' file if it already exists.
    daughterboard_number : str
        LinoSPAD2 daughterboard number.
    motherboard_number : str
        LinoSPAD2 motherboard (FPGA) number.
    firmware_version: str
        LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
        (block) are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).
    app_mask : bool, optional
        Switch for applying the mask for hot pixels. The default is True.
    include_offset : bool, optional
        Switch for applying offset calibration. The default is True.
    apply_calibration : bool, optional
        Switch for applying TDC and offset calibration. If set to 'True'
        while apply_offset_calibration is set to 'False', only the TDC
        calibration is applied. The default is True.
    chunksize : int, optional
        Number of files processed in each iteration. The default is 20.

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of
        'daughterboard_number', 'motherboard_number', and 'firmware_version'
        are accepted. The first error is raised so that the plot does not
        accidentally get rewritten in the case no clear input was given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(pixels, list) is False:
        raise TypeError(
            "'pixels' should be a list of integers or a list of two lists"
        )
    if isinstance(firmware_version, str) is False:
        raise TypeError(
            "'firmware_version' should be string, '2212s', '2212b' or '2208'"
        )
    if isinstance(rewrite, bool) is False:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(daughterboard_number, str) is False:
        raise TypeError("'daughterboard_number' should be string")

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
    )

    os.chdir(path)

    # Find all LinoSPAD2 data files
    files = sorted(glob.glob("*.dat"))
    # Get the resulting Feather file name based on the data files
    # found
    feather_file_name = files[0][:-4] + "-" + files[-1][:-4] + ".feather"
    # Construct absolute path to the Feather file
    feather_file = os.path.join(path, "delta_ts_data", feather_file_name)
    # Handle the rewrite parameter based on the file existence to avoid
    # accidental file overwritting
    utils.file_rewrite_handling(feather_file, rewrite)

    with multiprocessing.Manager() as manager:
        shared_result_queue = manager.Queue()
        shared_lock = manager.Lock()

        with multiprocessing.Pool() as pool:
            # Start the writer process
            writer_process = multiprocessing.Process(
                target=write_results_to_feather,
                args=(shared_result_queue, feather_file, shared_lock),
            )
            writer_process.start()

            # Create a partial function with fixed arguments for
            # process_file
            partial_process_file = functools.partial(
                calculate_timestamps_differences,
                result_queue=shared_result_queue,
                data_params=data_params,
            )

            # Start the multicore analysis of the files
            pool.map(partial_process_file, files, chunksize=chunksize)

            # Use tqdm to create a progress bar for the file processing
            # TODO Can't configure tqdm to update the progress bar
            # correctly while using chunksize in pool

            # with tqdm(total=len(files), desc="Processing files") as pbar:
            #     for _ in pool.imap_unordered(partial_process_file, files):
            #         pbar.update(1)

            # Signal the writer process that no more results will be
            # added to the queue
            shared_result_queue.put(None)
            writer_process.join()
