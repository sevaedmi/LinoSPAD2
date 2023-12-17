import glob
import os
import pickle

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
    # path_to_mask = os.path.realpath(__file__) + "/../.." + "/params/masks"
    path_to_mask = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "params",
        "masks",
    )
    os.chdir(path_to_mask)
    file_mask = glob.glob(
        "*{}_{}*".format(daughterboard_number, motherboard_number)
    )[0]
    mask = np.genfromtxt(file_mask).astype(int)
    os.chdir(path_to_back)

    return mask


def unpickle_plot(path: str, type: str, interactive: bool = True) -> None:
    """Unpickle and display a matplotlib figure.

    This function loads a previously pickled matplotlib figure from the
    specified path and displays it. Optionally, it can switch to an
    interactive backend.

    Parameters
    ----------
    path : str
        Path to the folder containing the pickled matplotlib figure.
    type : str
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
    if type not in ["sensor_population", "fits", "delta_t"]:
        raise TypeError(
            "Type should be 'sensor_population', 'fits', or 'delta_t'"
        )

    os.chdir(path)
    files = glob.glob("*.dat*")
    plot_name = files[0][:-4] + "-" + files[-1][:-4]

    os.chdir(f"results/{type}")

    if interactive:
        matplotlib.use("QtAgg")

    with open(f"{plot_name}.pickle", "rb") as file:
        fig = pickle.load(file)
        fig.show()


def gaussian(x, A, mu, sigma, bkg):
    """Gaussian function.

    Parameters
    ----------
    x : array-like
        The input data.
    A : float
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
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + bkg


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
            Optimal values for the parameters so that the sum of the squared residuals
            of `gaussian(x, *popt) - y` is minimized.
        - pcov : 2D array-like
            Estimated covariance of `popt`. The diagonal elements represent
            the variance of the parameter estimates.

    """
    # Initial guess for the parameters
    A_guess = np.max(y)
    mu_guess = x[np.argmax(y)]
    sigma_guess = np.std(x)
    bkg_guess = np.median(y)

    # Perform the curve fitting
    popt, pcov = curve_fit(
        gaussian, x, y, p0=[A_guess, mu_guess, sigma_guess, bkg_guess]
    )

    return popt, pcov
