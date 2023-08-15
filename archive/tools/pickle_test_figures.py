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
