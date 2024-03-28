import os
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyarrow import feather as ft

from LinoSPAD2.functions import utils


def sigma_of_count_spread_to_average(path, pixels: list):

    os.chdir(path)

    ft_file = glob("*.feather")[0]

    data = ft.read_feather(ft_file)

    data_cut = data[f"{pixels[0]},{pixels[1]}"]

    data_cut = data_cut[(data_cut > 20e3) & (data_cut < 40e3)]

    step = 12
    bins = np.arange(np.min(data_cut), np.max(data_cut), 17.857 * step)

    counts, bin_edges = np.histogram(data_cut, bins=bins)

    bin_centers = (bin_edges - 17.857 * step / 2)[1:]

    plt.rcParams.update({"font.size": 22})

    plt.figure(figsize=(10, 7))
    plt.step(bin_centers, counts)
    plt.title(f"Histogram of delta ts\nBin size is {bins[1] - bins[0]:.2f} ps")
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")

    counts_spread, bin_edges_spread = np.histogram(counts, bins=20)
    bin_centers_spread = (
        bin_edges_spread - (bin_edges_spread[1] - bin_edges_spread[0]) / 2
    )[1:]

    sns.jointplot(
        x=bin_centers, y=counts, height=10, marginal_kws=dict(bins=20)
    )
    plt.title("Histogram of delta ts with histograms of spread", fontsize=20)
    plt.xlabel(r"$\Delta$t [ps]", fontsize=20)
    plt.ylabel("# of coincidences [-]", fontsize=20)

    pars, covs = utils.fit_gaussian(bin_centers_spread, counts_spread)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.step(bin_centers_spread, counts_spread, label="Spread of counts")
    ax.plot(
        bin_centers_spread,
        utils.gaussian(bin_centers_spread, *pars),
        label="Fit",
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


def sigma_of_count_spread_to_average_from_ft_file(
    ft_file, pixels: list, step: int = 10, bins_sigma: int = 20
):

    path = os.path.dirname(ft_file)

    # feather_file_name = ft_file.split("\\")[-1].split(".")[0]

    os.chdir(path)

    data = ft.read_feather(ft_file)

    data_cut = data[f"{pixels[0]},{pixels[1]}"]

    data_cut = data_cut[(data_cut > 20e3) & (data_cut < 40e3)]

    bins = np.arange(np.min(data_cut), np.max(data_cut), 17.857 * step)

    counts, bin_edges = np.histogram(data_cut, bins=bins)

    bin_centers = (bin_edges - 17.857 * step / 2)[1:]

    plt.rcParams.update({"font.size": 22})

    plt.figure(figsize=(10, 7))
    plt.step(bin_centers, counts)
    plt.title(f"Histogram of delta ts\nBin size is {bins[1] - bins[0]:.2f} ps")
    plt.xlabel(r"$\Delta$t [ps]")
    plt.ylabel("# of coincidences [-]")

    counts_spread, bin_edges_spread = np.histogram(counts, bins=bins_sigma)
    bin_centers_spread = (
        bin_edges_spread - (bin_edges_spread[1] - bin_edges_spread[0]) / 2
    )[1:]

    sns.jointplot(
        x=bin_centers, y=counts, height=10, marginal_kws=dict(bins=bins_sigma)
    )
    plt.title("Histogram of delta ts with histograms of spread", fontsize=20)
    plt.xlabel(r"$\Delta$t [ps]", fontsize=20)
    plt.ylabel("# of coincidences [-]", fontsize=20)

    pars, covs = utils.fit_gaussian(bin_centers_spread, counts_spread)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.step(bin_centers_spread, counts_spread, label="Spread of counts")
    ax.plot(
        bin_centers_spread,
        utils.gaussian(bin_centers_spread, *pars),
        label="Fit",
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


path = r"D:\LinoSPAD2\Data\board_NL11\Prague\Halogen_HBT\Halogen_HBT"
ft_file = os.path.join(path, r"10247000-10247220.feather")
sigma_of_count_spread_to_average_from_ft_file(
    ft_file, pixels=[144, 171], step=10
)
