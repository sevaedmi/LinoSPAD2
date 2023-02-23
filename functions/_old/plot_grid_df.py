import os
import glob
from tqdm import tqdm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from functions import unpack as f_up
from functions.calc_diff import calc_diff_df


def plot_grid_df(path, pix, timestamps: int = 512):
    """
    This function plots a grid of timestamp differences for the given range of
    pixels. This function utilizes the pandas dataframes.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : list or array-like
        Pixel numbers for which the delta ts are calculated.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default is 512.

    Returns
    -------
    None.

    """

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, file in enumerate(DATA_FILES):
        data = f_up.unpack_binary_df(file, timestamps)

        fig, axes = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(24, 24))
        plt.suptitle("{} delta ts".format(file))

        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
                if w <= q:
                    continue
                deltas = calc_diff_df(
                    data[data.Pixel == pix[q]], data[data.Pixel == pix[w]]
                )

                if "Ne" and "540" in path:
                    chosen_color = "seagreen"
                elif "Ne" and "656" in path:
                    chosen_color = "orangered"
                elif "Ar" in path:
                    chosen_color = "mediumslateblue"
                else:
                    chosen_color = "salmon"
                try:
                    bins = np.arange(int(deltas.min()), int(deltas.max()), 17.857 * 2)
                except Exception:
                    continue
                sns.histplot(
                    ax=axes[q, w - 1],
                    x="Delta t",
                    data=deltas,
                    bins=bins,
                    color=chosen_color,
                )
                axes[q, w - 1].set_title(
                    "Pixels {pix1}-{pix2}".format(pix1=pix[q], pix2=pix[w])
                )
        try:
            os.chdir("results/delta_t")
        except FileNotFoundError:
            os.makedirs("results/delta_t")
            os.chdir("results/delta_t")
        fig.tight_layout()  # for perfect spacing between the plots
        plt.savefig("{name}_delta_t_grid_df.png".format(name=file))
        os.chdir("../..")
