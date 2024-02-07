    import os
    from glob import glob

    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt

    from LinoSPAD2.functions import plot_tmsp

    init_path = r"D:\LinoSPAD2\Data\board_NL11\Prague\rate_on_clock\#21"

    os.chdir(init_path)

    # rates = []
    freqs = [250, 200, 150, 100, 80, 50]
    
    folders = []
    
    for freq in freqs:
        folders.append(glob(f"{freq}*")[0])
    

    paths = [os.path.join(init_path, folder) for folder in folders]

    rates_dic = {}

    # for path in paths:
    #     plot_tmsp.plot_sensor_population(
    #         path,
    #         daughterboard_number="NL11",
    #         motherboard_number="#21",
    #         firmware_version="2212s",
    #         timestamps=300,
    #         include_offset=False,
    #         fit_peaks=True,
    #         show_fig=True,
    #         correct_pixel_addressing=True,
    #     )
    #     os.chdir(path)
    #     files = glob("*.dat")
    #     for file in files:
    #         plot_tmsp.collect_data_and_apply_mask(
    #             file,
    #             daughterboard_number="NL11",
    #             motherboard_number="#21",
    #             firmware_version="2212s",
    #             timestamps=300,
    #             include_offset=False,
    #             save_to_file=True,
    #         )

    #     plot_tmsp.plot_single_pix_hist(
    #         path,
    #         pixels=[12],
    #         daughterboard_number="NL11",
    #         motherboard_number="#33",
    #         firmware_version="2212s",
    #         timestamps=300,
    #         include_offset=False,
    #         fit_average=True,
    #     )

    # Collect processed data from txt and plot
    for i, freq in enumerate(freqs):
        folder_freq = glob(f"{freq}*")[0]
        
        os.chdir(os.path.join(init_path, folder_freq))
        files = glob("*.dat")
        rates_dic[f"{freq}"] = []
        os.chdir("senpop_data")

        for file in files:
            senpop_file = glob(f"*{file[:-4]}*.txt")[0]
            senpop_data = np.loadtxt(senpop_file)
            rate = np.max(senpop_data) / 500 * 250
            rates_dic[f"{freq}"].append(rate)
        os.chdir("../..")

    rates_df = pd.DataFrame.from_dict(rates_dic)

    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-Green
        "#17becf",  # Turquoise
        "#1a1a1a",  # Black
        "#ff9896",  # Light Red
        "#aec7e8",  # Light Blue
        "#ffbb78",  # Light Orange
        "#98df8a",  # Light Green
        "#c5b0d5",  # Light Purple
        "#c49c94",  # Light Brown
        "#f7b6d2",  # Light Pink
        "#c7c7c7",  # Light Gray
        "#dbdb8d",  # Light Yellow-Green
    ]


    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams.update({"font.size": 22})
    # for i, row in enumerate(rates_df):
    #     ax.plot(freqs, rates_df[f'{freq}'] / 1000, color=colors[i])
    #     ax.set_xlabel("External clock frequency [Hz]")
    #     ax.set_ylabel("Rate [kHz]")
    #     ax.set_title("From 250 to 50 Hz")
    for index, row in rates_df.iterrows():
        ax.plot(freqs, row, 'o')
    ax.set_xlabel("External clock frequency [Hz]")
    ax.set_ylabel("Rate [kHz]")
    title = ax.set_title("Both")
    path_to_save = os.path.join(init_path, title.get_text())
    plt.savefig(f"{path_to_save}.png")

    # Plot the average with error
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.rcParams.update({"font.size": 22})
    means = rates_df.mean(axis=0)
    errors = rates_df.std(axis=0)
    ax.errorbar(freqs, means, fmt="o", yerr=errors)
    ax.set_xlabel("External clock frequency [Hz]")
    ax.set_ylabel("Average rate [kHz]")
    title = ax.set_title("Both, average")
    path_to_save = os.path.join(init_path, title.get_text())
    plt.savefig(f"{path_to_save}.png")


    # os.chdir(r"D:\LinoSPAD2\Data\board_NL11\Prague\rate_on_clock\#21\250 Hz")
    # files = glob("*.dat")
    # senpop = plot_tmsp.collect_data_and_apply_mask(
    #     files,
    #     daughterboard_number="NL11",
    #     motherboard_number="#33",
    #     firmware_version="2212s",
    #     timestamps=300,
    #     include_offset=False,
    #     # correct_pixel_addressing=True,
    # )
    # plt.plot(senpop)
