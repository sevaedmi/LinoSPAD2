import os
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyarrow import feather as ft

from LinoSPAD2.functions import utils


def sigma_of_count_spread_to_average(
    path: str, pixels: list, step: int = 10, bins_sigma: int = 20
):
    """Plot and fit background spread from the feather file.

    Collect timestamp differences for the requested pixels from the
    given feather file, plot histogram of the background signal,
    plot a jointplot of the background with the spread of the
    background, fit the spread and calculate the ratio of the sigma of
    that spread to average background signal.

    Parameters
    ----------
    ft_file : str
        Feather file with timestamp differences.
    pixels : list
        Pixels which should be used for analysis.
    step : int, optional
        Multiplier of the average LinoSPAD2 TDC width of 17.857 ps that
        is used for histograms. The default is 10.
    bins_sigma : int, optional
        Number of bins used for plotting the histogram of the background
        spread. The default 20.
    """
    os.chdir(path)

    ft_file = glob("*.feather")[0]

    ft_file_name = ft_file.split(".")[0]

    data = ft.read_feather(ft_file)

    data_cut = data[f"{pixels[0]},{pixels[1]}"]

    # Cut the data from the background only; without the offset
    # calibration, the delta t peak rarely goes outside the 10 ns mark
    data_cut = data_cut[(data_cut > 20e3) & (data_cut < 40e3)]

    # Bins in units of 17.857 ps of the average LinoSPAD2 TDC bin width
    bins = np.arange(
        np.min(data_cut), np.max(data_cut), 2.5 / 140 * 1e3 * step
    )

    counts, bin_edges = np.histogram(data_cut, bins=bins)

    bin_centers = (bin_edges - 2.5 / 140 * 1e3 * step / 2)[1:]

    plt.rcParams.update({"font.size": 22})

    try:
        os.chdir("results/bckg_spread")
    except Exception:
        os.makedirs("results/bckg_spread")
        os.chdir("results/bckg_spread")

    # Background histogram
    plt.figure(figsize=(10, 7))
    plt.step(bin_centers, counts, color="tomato")
    plt.title(f"Histogram of delta ts\nBin size is {bins[1] - bins[0]:.2f} ps")
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.savefig(f"{ft_file_name}_bckg_hist.png")

    # Seaborn join histograms of background including the spread
    sns.jointplot(
        x=bin_centers, y=counts, height=10, marginal_kws=dict(bins=bins_sigma)
    )
    plt.title("Histogram of delta ts with histograms of spread", fontsize=20)
    plt.xlabel(r"$\Delta$t [ps]", fontsize=20)
    plt.ylabel("# of coincidences [-]", fontsize=20)
    plt.savefig(f"{ft_file_name}_bckg_hist_joint.png")

    # Histogram of the spread plus Gaussian fit
    counts_spread, bin_edges_spread = np.histogram(counts, bins=bins_sigma)
    bin_centers_spread = (
        bin_edges_spread - (bin_edges_spread[1] - bin_edges_spread[0]) / 2
    )[1:]

    pars, covs = utils.fit_gaussian(bin_centers_spread, counts_spread)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.step(
        bin_centers_spread,
        counts_spread,
        label="Spread of counts",
        color="darkorchid",
    )
    ax.plot(
        bin_centers_spread,
        utils.gaussian(bin_centers_spread, *pars),
        label="Fit",
        color="#cc8c32",
    )
    ax.set_title(
        f"Ratio of spread to average: {pars[2] / np.mean(counts) * 100:.1f} %"
    )
    ax.set_xlabel("Spread [-]")
    ax.set_ylabel("Counts [-]")
    ax.text(
        0.07,
        0.9,
        f"\u03C3={pars[2]:.2f}\u00B1{np.sqrt(covs[2,2]):.2f}",
        transform=ax.transAxes,
        fontsize=25,
        bbox=dict(
            facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )
    plt.savefig(f"{ft_file_name}_bckg_spread_hist_.png")


def sigma_of_count_spread_to_average_from_ft_file(
    ft_file: str, pixels: list, step: int = 10, bins_sigma: int = 20
):
    """Plot and fit background spread from the feather file.

    Collect timestamp differences for the requested pixels from the
    given feather file, plot histogram of the background signal,
    plot a jointplot of the background with the spread of the
    background, fit the spread and calculate the ratio of the sigma of
    that spread to average background signal. Only the feather file
    is required without the raw data and the feather file is chosen
    directly.

    Parameters
    ----------
    ft_file : str
        Feather file with timestamp differences.
    pixels : list
        Pixels which should be used for analysis.
    step : int, optional
        Multiplier of the average LinoSPAD2 TDC width of 17.857 ps that
        is used for histograms. The default is 10.
    bins_sigma : int, optional
        Number of bins used for plotting the histogram of the background
        spread. The default 20.
    """

    ft_file_name = ft_file.split(".")[0]

    path = os.path.dirname(ft_file)

    os.chdir(path)

    data = ft.read_feather(ft_file)

    data_cut = data[f"{pixels[0]},{pixels[1]}"]

    # Cut the data from the background only; without the offset
    # calibration, the delta t peak rarely goes outside the 10 ns mark
    data_cut = data_cut[(data_cut > 20e3) & (data_cut < 40e3)]

    # Bins in units of 17.857 ps of the average LinoSPAD2 TDC bin width
    bins = np.arange(
        np.min(data_cut), np.max(data_cut), 2.5 / 140 * 1e3 * step
    )

    counts, bin_edges = np.histogram(data_cut, bins=bins)

    bin_centers = (bin_edges - 2.5 / 140 * 1e3 * step / 2)[1:]

    plt.rcParams.update({"font.size": 22})

    # Background histogram
    plt.figure(figsize=(10, 7))
    plt.step(bin_centers, counts, color="tomato")
    plt.title(f"Histogram of delta ts\nBin size is {bins[1] - bins[0]:.2f} ps")
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")
    plt.savefig(f"{ft_file_name}_bckg_hist.png")

    # Seaborn join histograms of background including the spread
    sns.jointplot(
        x=bin_centers, y=counts, height=10, marginal_kws=dict(bins=bins_sigma)
    )
    plt.title("Histogram of delta ts with histograms of spread", fontsize=20)
    plt.xlabel(r"$\Delta$t [ps]", fontsize=20)
    plt.ylabel("# of coincidences [-]", fontsize=20)
    plt.savefig(f"{ft_file_name}_bckg_hist_joint.png")

    # Histogram of the spread plus Gaussian fit
    counts_spread, bin_edges_spread = np.histogram(counts, bins=bins_sigma)
    bin_centers_spread = (
        bin_edges_spread - (bin_edges_spread[1] - bin_edges_spread[0]) / 2
    )[1:]

    pars, covs = utils.fit_gaussian(bin_centers_spread, counts_spread)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.step(
        bin_centers_spread,
        counts_spread,
        label="Spread of counts",
        color="darkorchid",
    )
    ax.plot(
        bin_centers_spread,
        utils.gaussian(bin_centers_spread, *pars),
        label="Fit",
        color="#cc8c32",
    )
    ax.set_title(
        f"Ratio of spread to average: {pars[2] / np.mean(counts) * 100:.1f} %"
    )
    ax.set_xlabel("Spread [-]")
    ax.set_ylabel("Counts [-]")
    ax.text(
        0.59,
        0.9,
        f"\u03C3={pars[2]:.2f}\u00B1{np.sqrt(covs[2,2]):.2f}",
        transform=ax.transAxes,
        fontsize=25,
        bbox=dict(
            facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"
        ),
    )
    plt.savefig(f"{ft_file_name}_bckg_spread_hist_.png")


path = r"D:\LinoSPAD2\Data\board_NL11\Prague\Halogen_HBT\Halogen_HBT"
ft_file = os.path.join(path, r"10247000-10247220.feather")
sigma_of_count_spread_to_average_from_ft_file(
    ft_file, pixels=[144, 171], step=10
)
