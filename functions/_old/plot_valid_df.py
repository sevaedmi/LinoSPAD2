import glob
import os
from functions import unpack as f_up


def plot_valid_df(
    path, scale: str = "linear", timestamps: int = 512, show_fig: bool = False
):
    """
    Function for plotting the number of valid timestamps per pixel. Works
    with tidy dataframes.

    Parameters
    ----------
    path : str
        Path to the datafiles.
    scale : str, optional
        Use 'log' for logarithmic scale, leave empty for linear. Default is
        'linear'.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is 512.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.

    Returns
    -------
    None.

    """

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")
    for i, file in enumerate(DATA_FILES):
        data = f_up.unpack_binary_df(file, lines_of_data=timestamps)
        # count values: how many valid timestamps in each pixel
        valid = data["Pixel"].value_counts()
        # sort by index
        valid = valid.sort_index()
        fig = valid.plot(figsize=(11, 7), color="salmon", fontsize=20, marker="o")
        fig.set_title(
            "{name}\nmax count: {maxc}".format(name=file, maxc=valid.max()), fontsize=20
        )
        fig.set_xlabel("Pixel [-]", fontsize=20)
        fig.set_ylabel("Number of timestamps [-]", fontsize=20)

        if scale == "log":
            fig.set_yscale("log")
        try:
            os.chdir("results")
        except FileNotFoundError:
            os.makedirs("results")
            os.chdir("results")
            fig.savefig("{}.png".format(file))
            os.chdir("..")
