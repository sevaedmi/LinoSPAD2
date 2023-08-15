import glob
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sg
from scipy.optimize import curve_fit

from functions import unpack

path = "D:/LinoSPAD2/Data/board_A5/FW 2212 block/Spectrometer/Ar"
os.chdir(path)

files = glob.glob("*.dat*")

valid_per_pixel = np.zeros(256)

for i, file in enumerate(files):
    data = unpack.unpack_2212(file, board_number="A5", fw_ver="block", timestamps=100)

    for i in range(0, 256):
        a = np.array(data["{}".format(i)])
        valid_per_pixel[i] = valid_per_pixel[i] + len(np.where(a > 0)[0])

mask = [70, 205, 212, 95, 157, 165, 57, 123, 187, 118, 251]

for i in mask:
    valid_per_pixel[i] = 0

pixels = np.arange(0, 256, 1)
x_nm = 0.1056272 * pixels + 787.0255896

# plt.figure(figsize=(16, 10))
# plt.plot(valid_per_pixel, "o-", color="salmon")
# plt.figure(figsize(16, 10))
# plt.plot(x_nm, valid_per_pixel, "o-", color="teal")

peak_centers = [77, 131, 139, 221, 232]
peak_centers_nm = np.array(peak_centers) * 0.1056272 + 787.0255896


def gauss(x, A, x0, sigma, C):
    return A * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + C


sigma = 0.1
av_bkg = np.average(valid_per_pixel)


x_nm1 = np.copy(x_nm[50:100])
valid_per_pixel1 = np.copy(valid_per_pixel[50:100])

x_nm2 = np.copy(x_nm[100:170])
valid_per_pixel2 = np.copy(valid_per_pixel[100:170])
valid_per_pixel2[35:42] = 0

x_nm3 = np.copy(x_nm[100:170])
valid_per_pixel3 = np.copy(valid_per_pixel[100:170])
valid_per_pixel3[27:35] = 0

x_nm4 = np.copy(x_nm[150:])
valid_per_pixel4 = np.copy(valid_per_pixel[150:])
valid_per_pixel4[77:86] = 0

x_nm5 = np.copy(x_nm[150:])
valid_per_pixel5 = np.copy(valid_per_pixel[150:])
valid_per_pixel5[66:74] = 0


# ===========================================================================
par1, pcov1 = curve_fit(
    gauss,
    x_nm1,
    valid_per_pixel[50:100],
    p0=[max(valid_per_pixel1), peak_centers_nm[0], sigma, av_bkg],
)

perr1 = np.sqrt(np.diag(pcov1))
vis_er1 = par1[0] / par1[3] ** 2 * 100 * perr1[-1]
fit_plot1 = gauss(x_nm1, par1[0], par1[1], par1[2], par1[3])
# ===========================================================================

par2, pcov2 = curve_fit(
    gauss,
    x_nm2,
    valid_per_pixel[100:170],
    p0=[max(valid_per_pixel2), peak_centers_nm[1], sigma, av_bkg],
)

perr2 = np.sqrt(np.diag(pcov2))
vis_er2 = par2[0] / par2[3] ** 2 * 100 * perr2[-1]
fit_plot2 = gauss(x_nm2, par2[0], par2[1], par2[2], par2[3])
# ===========================================================================

par3, pcov3 = curve_fit(
    gauss,
    x_nm3,
    valid_per_pixel[100:170],
    p0=[max(valid_per_pixel3), peak_centers_nm[2], sigma, av_bkg],
)

perr3 = np.sqrt(np.diag(pcov3))
vis_er3 = par3[0] / par3[3] ** 2 * 100 * perr3[-1]
fit_plot3 = gauss(x_nm3, par3[0], par3[1], par3[2], par3[3])
# ===========================================================================

par4, pcov4 = curve_fit(
    gauss,
    x_nm4,
    valid_per_pixel[150:],
    p0=[max(valid_per_pixel4), peak_centers_nm[3], sigma, av_bkg],
)

perr4 = np.sqrt(np.diag(pcov4))
vis_er4 = par4[0] / par4[3] ** 2 * 100 * perr4[-1]
fit_plot4 = gauss(x_nm4, par4[0], par4[1], par4[2], par4[3])
# ===========================================================================

par5, pcov5 = curve_fit(
    gauss,
    x_nm5,
    valid_per_pixel[150:],
    p0=[max(valid_per_pixel5), peak_centers_nm[4], sigma, av_bkg],
)

perr5 = np.sqrt(np.diag(pcov5))
vis_er5 = par5[0] / par5[3] ** 2 * 100 * perr5[-1]
fit_plot5 = gauss(x_nm5, par5[0], par5[1], par5[2], par5[3])


plt.rcParams.update({"font.size": 16})
plt.figure(figsize=(16, 10))
plt.xlabel("Wavelength [nm]")
plt.ylabel("Intensity [-]")
plt.plot(x_nm, valid_per_pixel, "o-", color="salmon", label="Data")
plt.plot(
    x_nm1,
    fit_plot1,
    color="#028990",
    label="\n"
    "\u03C3={p1}\u00B1{pe1} nm\n"
    "\u03BC={p2}\u00B1{pe2} nm".format(
        p1=format(par1[2], ".3f"),
        p2=format(par1[1], ".3f"),
        pe1=format(perr1[2], ".3f"),
        pe2=format(perr1[1], ".3f"),
    ),
)
plt.plot(
    x_nm2,
    fit_plot2,
    color="#029d90",
    label="\n"
    "\u03C3={p1}\u00B1{pe1} nm\n"
    "\u03BC={p2}\u00B1{pe2} nm".format(
        p1=format(par2[2], ".3f"),
        p2=format(par2[1], ".3f"),
        pe1=format(perr2[2], ".3f"),
        pe2=format(perr2[1], ".3f"),
    ),
)
plt.plot(
    x_nm3,
    fit_plot3,
    color="#02b190",
    label="\n"
    "\u03C3={p1}\u00B1{pe1} nm\n"
    "\u03BC={p2}\u00B1{pe2} nm".format(
        p1=format(par3[2], ".3f"),
        p2=format(par3[1], ".3f"),
        pe1=format(perr3[2], ".3f"),
        pe2=format(perr3[1], ".3f"),
    ),
)
plt.plot(
    x_nm4,
    fit_plot4,
    color="#02c590",
    label="\n"
    "\u03C3={p1}\u00B1{pe1} nm\n"
    "\u03BC={p2}\u00B1{pe2} nm".format(
        p1=format(par4[2], ".3f"),
        p2=format(par4[1], ".3f"),
        pe1=format(perr4[2], ".3f"),
        pe2=format(perr4[1], ".3f"),
    ),
)
plt.plot(
    x_nm5,
    fit_plot5,
    color="#02d990",
    label="\n"
    "\u03C3={p1}\u00B1{pe1} nm\n"
    "\u03BC={p2}\u00B1{pe2} nm".format(
        p1=format(par5[2], ".3f"),
        p2=format(par5[1], ".3f"),
        pe1=format(perr5[2], ".3f"),
        pe2=format(perr5[1], ".3f"),
    ),
)
plt.legend()
