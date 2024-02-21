import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

a = np.arange(100)

fig = plt.figure(figsize=(10, 7))
plt.plot(a)

pickle.dump(fig, open("FigureObject.fig.pickle", "wb"))

# fix_o = pickle.load(open('FigureObject.fig.pickle', 'rb'))

os.chdir("C:/Users/bruce/Documents/Quantum astrometry/CT")

plot_name = "0000074836.dat_0000074846.dat"

with open("{}.pickle".format(plot_name), "rb") as file:
    figx = pickle.load(file)


### multiple unpickles while changing fir parameters ###


import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

path0 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_full_int"
path1 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_full_int"

path2 = (
    r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_80%"
)
path3 = (
    r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_70%"
)
path4 = (
    r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_60%"
)
path5 = (
    r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_50%"
)

path6 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_80%_int"
path7 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_70%_int"
path8 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_60%_int"
path9 = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_2-0m_50%_int"

paths = [path0, path1, path2, path3, path4, path5, path6, path7, path8, path9]

plt.close('all')


%matplotlib qt
for path in paths:
    os.chdir(os.path.join(path, "results/sensor_population"))
    pickle_file = glob("*.pickle")[0]


    with open(pickle_file, "rb") as file:
        figx = pickle.load(file)
        figx.set_size_inches(8, 6)
        plt.xlim(169,173)