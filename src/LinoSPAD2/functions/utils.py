"""Module of utility functions.

This module provides a set of functions used in other LinoSPAD2 functions,
covering tasks such as applying masks, unpickling and displaying
matplotlib figures, defining Gaussian functions, fitting them to data,
and transforming lists of pixels.

This file can also be imported as a module and contains the following
functions:

    * apply_mask - Apply a mask to the given data based on the
    daughterboard and motherboard numbers.

    * unpickle_plot - Unpickle and display a matplotlib figure based on
    the specified type.

    * gaussian - Gaussian function for curve fitting.

    * fit_gaussian - Fit Gaussian function to data and return optimal
    parameters and covariance.

    * pixel_list_transform - Transform a list of pixels into two separate
    lists based on input type.

    * file_rewrite_handling - based on the file name given and the
    boolean paramter 'rewrite', handles the file overwriting based on
    its' existence. Introduced for clearer code and modularity.
"""


import glob
import os
import pickle
import sys
import time

import matplotlib
import numpy as np
from scipy.optimize import curve_fit


def apply_mask(daughterboard_number: str, motherboard_number: str) -> None:
    """Apply mask to the given data.

    Parameters
    ----------
    data : np.ndarray
        The data array to which the mask will be applied.
    daughterboard_number : str
        The LinoSPAD2 daughterboard number.
    motherboard_number : str
        The LinoSPAD2 motherboard number.

    Returns
    -------
    None
    """
    path_to_back = os.getcwd()
    path_to_mask = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "params",
        "masks",
    )
    os.chdir(path_to_mask)
    file_mask = glob.glob(f"*{daughterboard_number}_{motherboard_number}*")[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    return mask


def unpickle_plot(path: str, plot_type: str, interactive: bool = True) -> None:
    """Unpickle and display a matplotlib figure.

    This function loads a previously pickled matplotlib figure from the
    specified path and displays it. Optionally, it can switch to an
    interactive backend.

    Parameters
    ----------
    path : str
        Path to the folder containing the pickled matplotlib figure.
    plot_type : str
        Type of the plot. Should be one of 'sensor_population', 'fits',
        or 'delta_t'.
    interactive : bool, optional
        Switch for making the plot interactive by using QtAgg matplotlib
        backend. The default is True.

    Raises
    ------
    TypeError
        Raised if the 'type' parameter is not one of 'sensor_population',
        'fits', or 'delta_t'.

    Returns
    -------
    None.

    """
    if plot_type not in ["sensor_population", "fits", "delta_t"]:
        raise TypeError(
            "Type should be 'sensor_population', 'fits', or 'delta_t'"
        )

    os.chdir(path)
    files = glob.glob("*.dat*")
    plot_name = files[0][:-4] + "-" + files[-1][:-4]

    os.chdir(f"results/{plot_type}")

    if interactive:
        matplotlib.use("QtAgg")

    with open(f"{plot_name}.pickle", "rb") as file:
        fig = pickle.load(file)
        fig.show()


def gaussian(x, amp, mu, sigma, bkg):
    """Gaussian function.

    Parameters
    ----------
    x : array-like
        The input data.
    amp : float
        Amplitude of the Gaussian.
    mu : float
        Mean (center) of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.
    bkg : float
        Background offset.

    Returns
    -------
    array-like
        The computed Gaussian values.

    """
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + bkg


def fit_gaussian(x, y):
    """Fit Gaussian function to data.

    Parameters
    ----------
    x : array-like
        The x-axis data.
    y : array-like
        The y-axis data.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - popt : array-like
            Optimal values for the parameters so that the sum of the
            squared residuals of `gaussian(x, *popt) - y` is minimized.
        - pcov : 2D array-like
            Estimated covariance of `popt`. The diagonal elements represent
            the variance of the parameter estimates.

    """
    # Initial guess for the parameters
    amp_guess = np.max(y)
    mu_guess = x[np.argmax(y)]
    # As std sometimes gives nonsense, 200 ps is added as balancing
    sigma_guess = min(np.std(x), 200)
    bkg_guess = np.median(y)

    # Perform the curve fitting
    popt, pcov = curve_fit(
        gaussian, x, y, p0=[amp_guess, mu_guess, sigma_guess, bkg_guess]
    )

    return popt, pcov


def pixel_list_transform(pixels: list):
    """Transform a list of pixels into two separate lists.

    Transform the given list of pixels into two separate lists,
    based on the input type (list of integers, of lists, or a mix of
    both).

    Parameters:
        pixels : list
            A list of pixels.

    Returns:
        list: A list of the left pixels.
        list: A list of the right pixels.
    """

    if isinstance(pixels[0], list) and isinstance(pixels[1], list) is True:
        pixels_left, pixels_right = sorted(pixels)
    elif isinstance(pixels[0], int) and isinstance(pixels[1], list) is True:
        pixels_left, pixels_right = sorted([[pixels[0]], pixels[1]])
    elif isinstance(pixels[0], list) and isinstance(pixels[1], int) is True:
        pixels_left, pixels_right = sorted([pixels[0], [pixels[1]]])
    elif isinstance(pixels[0], int) and isinstance(pixels[1], int) is True:
        pixels_left = pixels
        pixels_right = pixels

    return pixels_left, pixels_right


def file_rewrite_handling(file: str, rewrite: bool):
    """Handle file rewriting based on the 'rewrite' parameter.

    The function checks if the specified file exists in the
    'delta_ts_data' directory.
    If it exists and 'rewrite' is True, the file is deleted after a
    countdown (5 sec), this is a good time to stop execution if needed.
    If 'rewrite' is False and the file exists, the function exits with a
    system exit error message.

    Parameters
    ----------
    file : str
        The file path or name to check for existence and potentially
        rewrite.
    rewrite : bool
        If True, the function attempts to rewrite the file after
        deleting it.
        If False and the file already exists, the function exits with an
        error message.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If 'rewrite' is False and the file already exists, the function
        exits with an error message.

    """
    try:
        # os.chdir("delta_ts_data")
        if os.path.isfile(file):
            if rewrite is True:
                print(
                    "\n! ! ! Feather file with timestamps differences already "
                    "exists and will be rewritten ! ! !\n"
                )
                for i in range(5):
                    print(f"\n! ! ! Deleting the file in {5 - i} ! ! !\n")
                    time.sleep(1)
                os.remove(file)
            else:
                sys.exit(
                    "\n Feather file already exists, 'rewrite' set to"
                    "'False', exiting."
                )
        # os.chdir("..")
    except FileNotFoundError:
        pass
