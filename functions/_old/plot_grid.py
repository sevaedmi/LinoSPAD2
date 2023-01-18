""" One of the first iterations of the function for plotting a grid of
timestamps differences.

"""

def plot_grid(
    path,
    pix,
    timestamps: int = 512,
    range_left: float = -2.5e3,
    range_right: float = 2.5e3,
    show_fig: bool = False,
    same_y: bool = True,
):
    """
    Plots a grid of delta t for different pairs of pixels for the
    pixels in the given range. The output is saved in the "results/delta_t"
    folder. In the case the folder does not exist, it is created automatically.


    Parameters
    ----------
    path : str
        Path to the data file.
    pix : array-like
        Array of indices of pixels for analysis.
    timestamps : int, optional
        Number of timestamps per pixel per acquisition cycle. The default is
        512.
    range_left : float, optional
        Left limit of the range for which the timestamps differences should be
        calculated. Default is -2.5e3.
    range_right : float, optional
        Right limit of the range for which the timestamps differences should be
        calculated. Default is 2.5e3.
    show_fig : bool, optional
        Switch for showing the output figure. The default is False.
    same_y : bool, optional
        Switch for setting the same ylim for all plots in the grid. The
        default is True.

    Returns
    -------
    None.

    """

    # check if the figure should appear in a separate window or not at all
    if show_fig is True:
        plt.ion()
    else:
        plt.ioff()
    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, filename in enumerate(DATA_FILES):

        print(
            "=====================================================\n"
            "Plotting a delta t grid, Working on {}\n"
            "=====================================================\n".format(filename)
        )

        data = f_up.unpack_numpy(filename, timestamps)

        data_pix = np.zeros((len(pix), len(data[0])))

        for i, num1 in enumerate(pix):
            data_pix[i] = data[num1]
        plt.rcParams.update({"font.size": 22})
        if len(pix) > 2:
            fig, axs = plt.subplots(len(pix) - 1, len(pix) - 1, figsize=(20, 20))
        else:
            fig = plt.figure(figsize=(14, 14))
        # check if the y limits of all plots should be the same
        if same_y is True:
            y_max_all = 0
        print("\n> > > Calculating the timestamp differences < < <\n")
        for q in tqdm(range(len(pix)), desc="Minuend pixel   "):
            for w in tqdm(range(len(pix)), desc="Subtrahend pixel"):
                if w <= q:
                    continue
                data_pair = np.vstack((data_pix[q], data_pix[w]))

                delta_ts = cd(
                    data_pair,
                    timestamps=timestamps,
                    range_left=range_left,
                    range_right=range_right,
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
                    bins = np.arange(np.min(delta_ts), np.max(delta_ts), 17.857 * 2)
                except Exception:
                    print("Couldn't calculate bins: probably not enough delta ts.")
                    continue
                if len(pix) > 2:
                    axs[q][w - 1].set_xlabel("\u0394t [ps]")
                    axs[q][w - 1].set_ylabel("Timestamps [-]")
                    n, b, p = axs[q][w - 1].hist(
                        delta_ts, bins=bins, color=chosen_color
                    )
                else:
                    plt.xlabel("\u0394t [ps]")
                    plt.ylabel("Timestamps [-]")
                    n, b, p = plt.hist(delta_ts, bins=bins, color=chosen_color)
                # find position of the histogram peak
                try:
                    n_max = np.argmax(n)
                    arg_max = format((bins[n_max] + bins[n_max + 1]) / 2, ".2f")
                except Exception:
                    arg_max = None
                if same_y is True:
                    try:
                        y_max = np.max(n)
                    except ValueError:
                        y_max = 0
                        print("\nCould not find maximum y value\n")
                    if y_max_all < y_max:
                        y_max_all = y_max
                    if len(pix) > 2:
                        axs[q][w - 1].set_ylim(0, y_max + 4)
                    else:
                        plt.ylim(0, y_max + 4)
                if len(pix) > 2:
                    axs[q][w - 1].set_xlim(range_left - 100, range_right + 100)

                    axs[q][w - 1].set_title(
                        "Pixels {p1}-{p2}\nPeak position {pp}".format(
                            p1=pix[q], p2=pix[w], pp=arg_max
                        )
                    )
                else:
                    plt.xlim(range_left - 100, range_right + 100)
                    plt.title(
                        "Pixels {p1}-{p2}\nPeak position {pp}".format(
                            p1=pix[q], p2=pix[w], pp=arg_max
                        )
                    )
        if same_y is True:
            for q in range(len(pix)):
                for w in range(len(pix)):
                    if w <= q:
                        continue
                    if len(pix) > 2:
                        axs[q][w - 1].set_ylim(0, y_max_all + 10)
                    else:
                        plt.ylim(0, y_max_all + 10)
        try:
            os.chdir("results/delta_t")
        except FileNotFoundError:
            os.mkdir("results/delta_t")
            os.chdir("results/delta_t")
        fig.tight_layout()  # for perfect spacing between the plots
        plt.savefig("{name}_delta_t_grid.png".format(name=filename))
        os.chdir("../..")