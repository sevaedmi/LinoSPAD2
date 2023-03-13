import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

path = "C:/Users/bruce/Documents/Quantum astrometry"
os.chdir(path)
file = glob.glob("*TagsHistogram_SPDC_2.txt*")[0]

data = np.genfromtxt(file, delimiter=";")
data = data.T
data[0] = data[0] * 1e9

plt.figure(figsize=(16, 10))
plt.plot(data[0], data[1], label="data")


def gauss(x, A, x0, sigma):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


arg_max = data[0][np.where(data[1] == np.max(data[1]))[0]][0]

n = data[1][:50]
sigma = 1
b = data[0][:50]

par, covariance = curve_fit(gauss, b, n, p0=[max(n), arg_max, sigma])
fit_plot = gauss(b, par[0], par[1], par[2])

plt.plot(
    b[:50],
    fit_plot,
    label="fit\n\u03BC={p1} ps\n"
    "\u03C3={p2} ns".format(p1=format(par[1], ".2f"), p2=format(par[-1], ".2f")),
)
plt.legend(loc="best")
plt.xlabel("Time diff [-]")
plt.ylabel("Count [-]")
