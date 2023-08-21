def deltas_save(
    path,
    pix,
    rewrite: bool,
    board_number: str,
    fw_ver: str,
    timestamps: int = 512,
    delta_window: float = 50e3,
):
    """Calculate and save timestamp differences into .csv file.

    Unpacks data into a dictionary, calculates timestamp differences for
    the requested pixels and saves them into a .csv table. Works with
    firmware versions '2208' and '2212b' (block). The plot is saved
    in the 'results' folder, which is created (if it does not already
    exist) in the same folder where data are.

    Parameters
    ----------
    path : str
        Path to data files.
    pix : list
        List of pixel numbers for which the timestamp differences should
        be calculate and saved.
    rewrite : bool
        Switch for rewriting the csv file if it already exists.
    board_number : str
        The LinoSPAD2 daughterboard number.
    fw_ver: str
        LinoSPAD2 firmware version. Versions "2208" and "2212b (block)"
        are recognized.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    delta_window : float, optional
        Size of a window to which timestamp differences are compared.
        Differences in that window are saved. The default is 50e3 (50 ns).

    Raises
    ------
    TypeError
        Only boolean values of 'rewrite' and string values of 'fw_ver'
        are accepted. First error is raised so that the plot does not
        accidentally gets rewritten in the case no clear input was
        given.

    Returns
    -------
    None.
    """
    # parameter type check
    if isinstance(fw_ver, str) is not True:
        raise TypeError("'fw_ver' should be string, '2212b' or '2208'")
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean")
    if isinstance(board_number, str) is not True:
        raise TypeError(
            "'board_number' should be string, either 'NL11' or 'A5'"
        )
    os.chdir(path)

    files_all = glob.glob("*.dat*")

    out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

    # check if csv file exists and if it should be rewrited
    try:
        os.chdir("delta_ts_data")
        if os.path.isfile("{name}.csv".format(name=out_file_name)):
            if rewrite is True:
                print(
                    "\n! ! ! csv file with timestamps differences already "
                    "exists and will be rewritten ! ! !\n"
                )
                for i in range(5):
                    print(
                        "\n! ! ! Deleting the file in {} ! ! !\n".format(5 - i)
                    )
                    time.sleep(1)
                os.remove("{}.csv".format(out_file_name))
            else:
                sys.exit(
                    "\n csv file already exists, 'rewrite' set to"
                    "'False', exiting."
                )
        os.chdir("..")
    except FileNotFoundError:
        pass

    # Collect the data for the required pixels
    print(
        "\n> > > Collecting data for delta t plot for the requested "
        "pixels and saving it to .csv in a cycle < < <\n"
    )
    if fw_ver == "2208":
        for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
            file = files_all[i]

            # Prepare a dictionary for output
            deltas_all = {}
            for q in pix:
                for w in pix:
                    if w <= q:
                        continue
                    deltas_all["{},{}".format(q, w)] = []

            # Unpack data for the requested pixels into dictionary
            data = f_up.unpack_numpy_dict(
                file, board_number=board_number, timestamps=timestamps, pix=pix
            )

            # Calculate and collect timestamp differences
            for q in pix:
                for w in pix:
                    if w <= q:
                        continue

                    # Follows the cycle number in the first array
                    cycle = 0
                    # Follows the cycle number in the second array
                    cyc2 = np.argwhere(data["{}".format(w)] < 0)
                    cyc2 = np.insert(cyc2, 0, -1)
                    for i, tmsp1 in enumerate(data["{}".format(q)]):
                        if tmsp1 == -2:
                            cycle += 1
                            continue
                        deltas = (
                            data["{}".format(w)][
                                cyc2[cycle] + 1 : cyc2[cycle + 1]
                            ]
                            - tmsp1
                        )
                        # Collect deltas in the requested window
                        ind = np.where(np.abs(deltas) < delta_window)[0]
                        deltas_all["{},{}".format(q, w)].extend(deltas[ind])

            # Save data as a .csv file
            data_for_plot_df = pd.DataFrame.from_dict(
                deltas_all, orient="index"
            )
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("delta_ts_data")
            except FileNotFoundError:
                os.mkdir("delta_ts_data")
                os.chdir("delta_ts_data")
            csv_file = glob.glob("*{}*".format(out_file_name))
            # create for first file, append for all next ones
            if csv_file != []:
                data_for_plot_df.to_csv(
                    "{}.csv".format(out_file_name), mode="a", header=False
                )
            else:
                data_for_plot_df.to_csv("{}.csv".format(out_file_name))
            os.chdir("..")
    elif fw_ver == "2212b":
        # for transforming pixel number into TDC number + pixel
        # coordinates in that TDC
        pix_coor = np.arange(256).reshape(64, 4)

        for i in tqdm(range(ceil(len(files_all))), desc="Collecting data"):
            file = files_all[i]

            # Prepare a dictionary for output
            deltas_all = {}

            # Unpack data for the requested pixels into dictionary
            data_all = f_up.unpack_bin(file, board_number, timestamps)

            # Calculate and collect timestamp differences
            for q in pix:
                for w in pix:
                    if w <= q:
                        continue
                    deltas_all["{},{}".format(q, w)] = []
                    # find end of cycles
                    cycler = np.argwhere(data_all[0].T[0] == -2)
                    cycler = np.insert(cycler, 0, 0)
                    # first pixel in the pair
                    tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
                    pix1 = np.where(data_all[tdc1].T[0] == pix_c1)[0]
                    # second pixel in the pair
                    tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
                    pix2 = np.where(data_all[tdc2].T[0] == pix_c2)[0]
                    # get timestamp for both pixels in the given cycle
                    for cyc in range(len(cycler) - 1):
                        pix1_ = pix1[
                            np.logical_and(
                                pix1 > cycler[cyc], pix1 < cycler[cyc + 1]
                            )
                        ]
                        if not np.any(pix1_):
                            continue
                        pix2_ = pix2[
                            np.logical_and(
                                pix2 > cycler[cyc], pix2 < cycler[cyc + 1]
                            )
                        ]
                        if not np.any(pix2_):
                            continue
                        # calculate delta t
                        tmsp1 = data_all[tdc1].T[1][
                            pix1_[np.where(data_all[tdc1].T[1][pix1_] > 0)[0]]
                        ]
                        tmsp2 = data_all[tdc2].T[1][
                            pix2_[np.where(data_all[tdc2].T[1][pix2_] > 0)[0]]
                        ]
                        for t1 in tmsp1:
                            deltas = tmsp2 - t1
                            ind = np.where(np.abs(deltas) < delta_window)[0]
                            deltas_all["{},{}".format(q, w)].extend(
                                deltas[ind]
                            )

            # Save data as a .csv file in a cycle so data is not lost
            # in the case of failure close to the end
            data_for_plot_df = pd.DataFrame.from_dict(
                deltas_all, orient="index"
            )
            del deltas_all
            data_for_plot_df = data_for_plot_df.T
            try:
                os.chdir("delta_ts_data")
            except FileNotFoundError:
                os.mkdir("delta_ts_data")
                os.chdir("delta_ts_data")
            csv_file = glob.glob("*{}.csv*".format(out_file_name))
            if csv_file != []:
                data_for_plot_df.to_csv(
                    "{}.csv".format(out_file_name),
                    mode="a",
                    index=False,
                    header=False,
                )
            else:
                data_for_plot_df.to_csv(
                    "{}.csv".format(out_file_name), index=False
                )
            os.chdir("..")
    print(
        "\n> > > Timestamp differences are saved as {file}.csv in"
        "{path} < < <".format(
            file=out_file_name,
            path=path + "delta_ts_data",
        )
    )
