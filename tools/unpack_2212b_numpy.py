import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


def deltas_save_2212_numpy(
    path, board_number: str, pixels, rewrite: bool, timestamps: int = 512
):
    # parameter type check
    if isinstance(board_number, str) is not True:
        raise TypeError("'board_number' should be string, 'NL11' or 'A5'")
    if isinstance(rewrite, bool) is not True:
        raise TypeError("'rewrite' should be boolean")
    os.chdir(path)

    files = glob.glob("*.dat*")

    out_file_name = files[0][:-4] + "-" + files[-1][:-4]

    # check if csv file exists and if it should be rewrited
    try:
        os.chdir("delta_ts_data")
        if os.path.isfile("{name}.csv".format(name=out_file_name)):
            if rewrite is True:
                print("\n! ! ! csv file already exists and will be rewritten. ! ! !\n")
                for i in range(5):
                    print("\n! ! ! Deleting the file in {} ! ! !\n".format(i))
                    time.sleep(1)
                os.remove("{}.csv".format(out_file_name))
            else:
                # print("\n> > > Plot already exists. < < <\n")
                sys.exit(
                    "\n csv file already exists, 'rewrite' set to 'False', exiting."
                )
        os.chdir("..")
    except FileNotFoundError:
        pass

    for i in tqdm(range(len(files)), desc="Going through files, collecting delta ts."):
        # unpack binary data
        rawFile = np.fromfile(files[i], dtype=np.uint32)
        # timestamps are lower 28 bits
        data_t = (rawFile & 0xFFFFFFF).astype(np.longlong) * 17.857
        # pix adress in the given TDC is 2 bits above timestamp
        data_p = ((rawFile >> 28) & 0x3).astype(int)
        data_t[np.where(rawFile < 0x80000000)] = -1
        cycles = int(len(data_t) / timestamps / 65)
        # transofrm into matrix 65 by cycles*timestamps
        data_matrix_p = (
            data_p.reshape(cycles, 65, timestamps)
            .transpose((1, 0, 2))
            .reshape(65, timestamps * cycles)
        )

        data_matrix_t = (
            data_t.reshape(cycles, 65, timestamps)
            .transpose((1, 0, 2))
            .reshape(65, timestamps * cycles)
        )
        # cut the 65th TDC that does not hold any actual data from pixels
        data_matrix_p = data_matrix_p[:-1]
        data_matrix_t = data_matrix_t[:-1]
        # insert '-2' at the end of each cycle
        data_matrix_p = np.insert(
            data_matrix_p,
            np.linspace(timestamps, cycles * timestamps, cycles).astype(int),
            -2,
            1,
        )

        data_matrix_t = np.insert(
            data_matrix_t,
            np.linspace(timestamps, cycles * timestamps, cycles).astype(int),
            -2,
            1,
        )
        # combine both matrices into a single one, where each cell holds pix coordinates
        # in the TDC and the timestamp
        data_all = np.stack((data_matrix_p, data_matrix_t), axis=2)
        # for transforming pixel number into TDC number + pixel coordinates in that TDC
        pix_coor = np.arange(256).reshape(64, 4)

        # Prepare a dictionary for output
        deltas_out = {}
        for q in pixels:
            for w in pixels:
                if w <= q:
                    continue
                deltas_out["{},{}".format(q, w)] = []

        for q in pixels:
            for w in pixels:
                if w <= q:
                    continue
                # find end of cycles
                cycler = np.argwhere(data_all[0].T[0] == -2)
                cycler = np.insert(cycler, 0, 0)
                # first pixel in the pair
                tdc1, pix1 = np.argwhere(pix_coor == q)[0]
                pix1 = np.where(data_all[tdc1].T[0] == pix1)[0]
                # second pixel in the pair
                tdc2, pix2 = np.argwhere(pix_coor == w)[0]
                pix2 = np.where(data_all[tdc2].T[0] == pix2)[0]
                # get timestamp for both pixels in the given cycle
                for i in range(len(cycler) - 1):
                    pix1_ = pix1[np.logical_and(pix1 > cycler[i], pix1 < cycler[i + 1])]
                    if not np.any(pix1_):
                        continue
                    pix2_ = pix2[np.logical_and(pix2 > cycler[i], pix2 < cycler[i + 1])]
                    if not np.any(pix2_):
                        continue
                    # calculate delta t
                    tmsp1_ = data_all[tdc1].T[1][pix1_]
                    tmsp2_ = data_all[tdc2].T[1][pix2_]
                    for t1 in tmsp1_:
                        deltas = tmsp2_ - t1
                        ind = np.where(np.abs(deltas) < 50e3)[0]
                        deltas_out["{},{}".format(q, w)].extend(deltas[ind])

        # Save data as a .csv file
        data_for_plot_df = pd.DataFrame.from_dict(deltas_out, orient="index")
        data_for_plot_df = data_for_plot_df.T
        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            os.mkdir("delta_ts_data")
            os.chdir("delta_ts_data")
        csv_file = glob.glob("*{}*".format(out_file_name))
        if csv_file != []:
            data_for_plot_df.to_csv(
                "{}.csv".format(out_file_name), mode="a", index=False
            )
        else:
            data_for_plot_df.to_csv("{}.csv".format(out_file_name))
        os.chdir("..")


path = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_703"

deltas_save_2212_numpy(
    path, board_number="A5", pixels=[3, 45], rewrite=True, timestamps=1000
)


# =============================================================================
#
# =============================================================================


import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/Ne_703/delta_ts_data/0000014157-0000014256.csv"
data = pd.read_csv(file)

for column in data.columns:
    a = data[data[column].str.contains(",")].index
    data = pd.DataFrame(data[column].drop(labels=np.array(a), axis=0))

data_to_plot = np.array(data["3,45"]).astype(float)
data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot < -20e3))
data_to_plot = np.delete(data_to_plot, np.argwhere(data_to_plot > 20e3))
bins = np.linspace(
    np.min(data_to_plot),
    np.max(data_to_plot),
    100,
)

plt.figure()
plt.hist(data_to_plot, bins=bins)
plt.show()
