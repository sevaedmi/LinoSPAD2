import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from LinoSPAD2.functions import cross_talk, plot_tmsp

path33 = r"D:\LinoSPAD2\Data\board_NL11\Prague\DCR\#33"
path21 = r"D:\LinoSPAD2\Data\board_NL11\Prague\DCR\#21"

dcr33 = cross_talk.collect_dcr_by_file(
    path33,
    daughterboard_number="NL11",
    motherboard_number="#33",
    firmware_version="2212s",
    timestamps=1000,
)

file_path = "DCR_#33.pkl"

with open(file_path, "wb") as file:
    pickle.dump(dcr33, file)

dcr21 = cross_talk.collect_dcr_by_file(
    path21,
    daughterboard_number="NL11",
    motherboard_number="#21",
    firmware_version="2212s",
    timestamps=1000,
)

file_path = "DCR_#21.pkl"

with open(file_path, "wb") as file:
    pickle.dump(dcr21, file)


path = r"D:\LinoSPAD2\Data\board_NL11\Prague\DCR"

plot_tmsp.plot_sensor_population_full_sensor(
    path,
    daughterboard_number="NL11",
    motherboard_number1="#33",
    motherboard_number2="#21",
    firmware_version="2212s",
    timestamps=1000,
    include_offset=False,
    app_mask=False,
)

##################
hot_pixels_21 = [5, 7, 35, 100, 121, 189, 198, 220, 225, 228, 229, 247]
hot_pixels_33 = [15, 50, 52, 66, 93, 98, 109, 122, 210, 231, 236]

# 33
plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 20})
x_axis = [x * 64 / 60 for x in range(0, 648)]
plt.xlabel("Time [min]")
plt.ylabel("Normalized DCR [-]")
plt.plot(
    x_axis,
    [x[15] for x in dcr33] / np.max([x[15] for x in dcr33]),
    "--",
    label="Hot pixel 15",
)
plt.plot(
    x_axis,
    [x[50] for x in dcr33] / np.max([x[50] for x in dcr33]),
    "--",
    label="Hot pixel 50",
)
plt.plot(
    x_axis,
    [x[122] for x in dcr33] / np.max([x[122] for x in dcr33]),
    "--",
    label="Hot pixel 121",
)
plt.plot(
    x_axis,
    [x[210] for x in dcr33] / np.max([x[210] for x in dcr33]),
    "--",
    label="Hot pixel 220",
)
plt.legend()

plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 20})
x_axis = [x * 64 / 60 for x in range(0, 648)]
plt.xlabel("Time [min]")
plt.ylabel("Normalized DCR [-]")
plt.plot(x_axis, [x[15] for x in dcr33], "--", label="Hot pixel 15")
plt.plot(x_axis, [x[50] for x in dcr33], "--", label="Hot pixel 50")
plt.plot(x_axis, [x[122] for x in dcr33], "--", label="Hot pixel 122")
plt.plot(x_axis, [x[210] for x in dcr33], "--", label="Hot pixel 210")
plt.legend()

# 21
plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 20})
x_axis = [x * 64 / 60 for x in range(0, 600)]
plt.xlabel("Time [min]")
plt.ylabel("Normalized DCR [-]")
plt.plot(
    x_axis,
    [x[5] for x in dcr] / np.max([x[5] for x in dcr]),
    "--",
    label="Hot pixel 5",
)
plt.plot(
    x_axis,
    [x[35] for x in dcr] / np.max([x[35] for x in dcr]),
    "--",
    label="Hot pixel 35",
)
plt.plot(
    x_axis,
    [x[100] for x in dcr] / np.max([x[100] for x in dcr]),
    "--",
    label="Hot pixel 100",
)
plt.plot(
    x_axis,
    [x[228] for x in dcr] / np.max([x[228] for x in dcr]),
    "--",
    label="Hot pixel 228",
)
plt.legend()
