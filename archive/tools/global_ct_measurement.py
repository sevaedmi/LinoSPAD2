import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from pyarrow import feather as ft

from LinoSPAD2.functions import cross_talk, delta_t, plot_tmsp

path = r"/media/sj/King4TB/LS2_Data/CT"

os.chdir(path)
files = glob("*.dat")

# plot_tmsp.collect_data_and_apply_mask(
#     files,
#     daughterboard_number="NL11",
#     motherboard_number="#21",
#     firmware_version="2212s",
#     timestamps=1000,
#     include_offset=False,
#     app_mask=False,
#     save_to_file=True,
# )

# plot_tmsp.plot_sensor_population(
#     path,
#     daughterboard_number="NL11",
#     motherboard_number="#21",
#     firmware_version="2212s",
#     timestamps=1000,
#     include_offset=False,
#     # correct_pixel_addressing=True,
#     app_mask=False,
#     show_fig=True,
#     # fit_peaks=True
# )

# hot_pixels = [100, 119, 134, 151, 155, 220]
# hot_pixels = [5, 7, 35, 121, 225, 228, 247] # 7 is problematic
hot_pixels = [5, 35, 121, 225, 228]  # on right
# hot_pixels = [35, 121, 225, 228, 247] # on left
hot_pixels_plus_20 = [[x + i for i in range(0, 21)] for x in hot_pixels]
hot_pixels_minus_20 = [[x - i for i in range(0, 21)] for x in hot_pixels]

# pixels = [x for x in range(119, 140)]
pixels = list(range(220, 199, -1))

os.chdir(os.path.join(path, "senpop_data"))
senop_data_txt = glob("*.txt")[0]
senpop = np.genfromtxt(senop_data_txt)


CT_all = []
CT_err_all = []


for pixels in hot_pixels_plus_20:
    # tmsps = cross_talk.collect_cross_talk_detailed(
    #     path,
    #     pixels=pixels,
    #     daughterboard_number="NL11",
    #     motherboard_number="#21",
    #     firmware_version="2212s",
    #     timestamps=1000,
    #     rewrite=True,
    #     include_offset=False,
    #     step=10,
    # )

    # CT, CT_err = cross_talk.plot_cross_talk_peaks(
    #     path,
    #     pixels=pixels,
    #     step=14,
    #     window=100e3,
    #     senpop=senpop,
    #     # pix_on_left=True,
    # )
    cross_talk.plot_cross_talk_grid(
        path,
        pixels=pixels,
        step=14,
        window=100e3,
        senpop=senpop,
        # pix_on_left=True,
    )

    CT_all.append(CT)
    CT_err_all.append(CT_err)

os.chdir(r"/home/sj/LS2_Data/CT_#21")

for i, (CT, CT_err) in enumerate(zip(CT_all, CT_err_all)):
    if CT == {} or CT_err == {}:
        continue

    differences = []
    keys = CT.keys()

    for key in keys:
        # Convert the key to a tuple
        key_tuple = eval(key)
        # Extract the difference and append it to the list
        differences.append(key_tuple[1] - key_tuple[0])

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({"font.size": 22})
    plt.errorbar(
        differences,
        list(CT.values()),
        list(CT_err.values()),
        fmt=".",
        color="indianred",
    )
    aggressor_pix = int(list(CT.keys())[0].split(",")[0].split("(")[1])
    plt.title(f"Cross-talk probability for aggressor pixel {aggressor_pix}")
    plt.xlabel("Distance in pixels [-]")
    plt.ylabel("Cross-talk probability [%]")
    plt.savefig(f"Cross-talk_aggressor_pixel_{aggressor_pix}_onright.png")


final_result = {key: [] for key in range(1, 21)}
final_result_averages = {key: [] for key in range(1, 21)}
# final_result = {key: [] for key in range(-20, 0)}
# final_result_averages = {key: [] for key in range(-20, 0)}

for i in range(len(CT_all)):
    if CT_all[i] == {} or CT_err_all[i] == {}:
        continue

    for key in CT_all[i].keys():
        key_tuple = eval(key)
        key_difference = key_tuple[1] - key_tuple[0]
        final_result[key_difference].append(
            (CT_all[i][key], CT_err_all[i][key])
        )

for key in final_result.keys():
    value = np.average([x[0] for x in final_result[key]])
    error = np.sqrt(np.sum([x[1] ** 2 for x in final_result[key]])) / len(
        final_result[key]
    )
    final_result_averages[key].append((value, error))

# %matplotlib qt
plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 22})
plt.title("Average cross-talk probability")
plt.xlabel("Distance in pixels [-]")
plt.ylabel("Cross-talk probability [%]")
plt.errorbar(
    final_result_averages.keys(),
    [x[0][0] for x in final_result_averages.values()],
    yerr=[x[0][1] for x in final_result_averages.values()],
    fmt=".",
    color="darkred",
)
plt.savefig(f"Average_cross-talk_onright.png")

###############################################################################

on_left = {
    -20: [(2.6851745989756856e-05, 1.191887353187704e-05)],
    -19: [(2.8457791481878805e-05, 1.3824792975464233e-05)],
    -18: [(3.526698649615147e-05, 1.2243372684836011e-05)],
    -17: [(3.165683891537275e-05, 1.2332695542078395e-05)],
    -16: [(3.4931039338717954e-05, 1.41856235513107e-05)],
    -15: [(3.0253850208953465e-05, 2.379359361921644e-05)],
    -14: [(2.6602210104826465e-05, 1.0933131324668037e-05)],
    -13: [(6.0510302973305185e-05, 1.5668882063039332e-05)],
    -12: [(3.120469539814469e-05, 1.482443709555862e-05)],
    -11: [(2.759748481040062e-05, 9.516737297014157e-06)],
    -10: [(2.0942593083060074e-05, 7.572604049122653e-06)],
    -9: [(7.2411112005147334e-06, 3.830366736641884e-06)],
    -8: [(1.678831472507417e-05, 1.0628004118501952e-05)],
    -7: [(5.131359938476538e-05, 1.3663956592333516e-05)],
    -6: [(7.465137137859563e-05, 1.6444980408916293e-05)],
    -5: [(0.00010335156289071625, 2.272164348900739e-05)],
    -4: [(0.0006222476404492227, 5.1616150800406865e-05)],
    -3: [(0.001811288097610038, 0.00010474256806584456)],
    -2: [(0.012649349273195496, 0.00022276478781443952)],
    -1: [(0.2374147229185738, 0.0009526839405427923)],
}

on_left1 = {
    -20: [(1.3031119476565856e-05, 4.71175889360206e-06)],
    -19: [(2.2678399772435142e-05, 9.152576380591689e-06)],
    -18: [(4.2413699704557344e-05, 8.755903358094617e-06)],
    -17: [(2.726393672038891e-05, 8.024271202089667e-06)],
    -16: [(1.4842195468857562e-05, 6.308871258703216e-06)],
    -15: [(1.9538914847776112e-05, 8.2443141428608e-06)],
    -14: [(2.7890692129464648e-05, 8.908038261279422e-06)],
    -13: [(1.1258960080630815e-05, 8.70203742727661e-06)],
    -12: [(3.0046749068633835e-05, 7.065944720748718e-06)],
    -11: [(3.45087959111318e-05, 6.993410943637765e-06)],
    -10: [(2.080764484688431e-05, 6.8356922669604246e-06)],
    -9: [(2.720367410757538e-05, 8.03827752000691e-06)],
    -8: [(4.532326962292327e-05, 9.957517338879892e-06)],
    -7: [(7.440013640269966e-05, 1.1365504749425503e-05)],
    -6: [(0.0001057602844775044, 1.2924112823304803e-05)],
    -5: [(0.00022168637637653955, 1.689110155673192e-05)],
    -4: [(0.0005351985196775693, 2.5201090235554012e-05)],
    -3: [(0.0019139055237761785, 4.487189245615414e-05)],
    -2: [(0.012869235877761198, 0.00011902929036869342)],
    -1: [(0.2649956290844796, 0.0005377546200769003)],
}

on_right = {
    1: [(0.21526068519188923, 0.0009653034034226792)],
    2: [(0.011777635034479315, 0.0002020253369360976)],
    3: [(0.0018447392403180814, 8.88268914339437e-05)],
    4: [(0.000576040809329607, 4.968470251857478e-05)],
    5: [(0.00018863358545814391, 2.6674697456377102e-05)],
    6: [(7.133171033703032e-05, 2.5914300639599925e-05)],
    7: [(5.8522605310669454e-05, 1.5697239005182266e-05)],
    8: [(5.4711165895217206e-05, 7.821847164521014e-05)],
    9: [(2.616291599904461e-05, 1.1014288268243004e-05)],
    10: [(2.990275555119951e-05, 9.682728949279304e-06)],
    11: [(4.2168274992883194e-05, 1.6148282297825875e-05)],
    12: [(6.991853615168813e-05, 1.8948955654291743e-05)],
    13: [(5.033480255575438e-05, 1.8781025434328653e-05)],
    14: [(3.10283606587451e-05, 1.16712460194768e-05)],
    15: [(2.6872470706078545e-05, 1.0685785884778466e-05)],
    16: [(3.976878861180013e-05, 1.2029970183428196e-05)],
    17: [(1.1836249324112396e-05, 7.567037569173731e-06)],
    18: [(5.649036721289217e-05, 1.640539771376586e-05)],
    19: [(5.100662797512533e-06, 1.8835445661858957e-06)],
    20: [(4.66471906144401e-05, 2.300932814036917e-05)],
}

on_right1 = {
    1: [(0.16868568422969657, 0.0003763526332028734)],
    2: [(0.01105654489388786, 9.371098344249814e-05)],
    3: [(0.0016444492514385842, 4.1502462895616244e-05)],
    4: [(0.0005468767109693682, 2.5728650792905275e-05)],
    5: [(0.00019956067922930035, 1.577751875689218e-05)],
    6: [(9.554802077626923e-05, 1.235844513285423e-05)],
    7: [(5.8904325950689125e-05, 9.723824825765975e-06)],
    8: [(3.2414943555045286e-05, 6.9625320431798566e-06)],
    9: [(1.1274930096375444e-05, 5.947278237728643e-06)],
    10: [(2.967879585582966e-05, 7.430156712840182e-06)],
    11: [(7.98956994061042e-06, 5.509997493162243e-06)],
    12: [(2.0907649098338472e-05, 7.726005470483721e-06)],
    13: [(2.8201687703413736e-05, 6.905586437321009e-06)],
    14: [(1.2213203645920027e-05, 6.862636644839544e-06)],
    15: [(3.787459941393153e-05, 6.954280592340361e-06)],
    16: [(4.6565827746531195e-05, 8.08014861820144e-06)],
    17: [(1.6095729406328625e-05, 9.171758007635727e-06)],
    18: [(3.464075043976942e-05, 6.510664122147179e-06)],
    19: [(1.753791753365583e-05, 6.773596671926013e-06)],
    20: [(1.789384350213515e-05, 5.527221953180719e-06)],
}

on_both_average = {key: [] for key in range(1, 21)}
for key in on_left.keys():
    ct_value_average = (
        on_left[key][0][0]
        + on_right[np.abs(key)][0][0]
        + on_left1[key][0][0]
        + on_right1[np.abs(key)][0][0]
    ) / 4
    ct_error_average = (
        np.sqrt(
            on_left[key][0][1] ** 2
            + on_right[np.abs(key)][0][1] ** 2
            + on_left1[key][0][1] ** 2
            + on_right1[np.abs(key)][0][1] ** 2
        )
        / 4
    )
    on_both_average[np.abs(key)] = (ct_value_average, ct_error_average)


plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 22})
plt.title("Average cross-talk probability")
plt.xlabel("Distance in pixels [-]")
plt.ylabel("Cross-talk probability [%]")
plt.yscale("log")
plt.errorbar(
    on_both_average.keys(),
    [x[0] for x in on_both_average.values()],
    yerr=[x[1] for x in on_both_average.values()],
    fmt=".",
    color="darkred",
)
plt.tight_layout()
plt.savefig(f"Average_cross-talk.png")

from math import exp

from scipy.optimize import curve_fit


def exponen(x, a):
    return a**x


def exponen(x):
    return exp(-0.0025 / x)


distance = list(on_both_average.keys())
CT_measured = [
    0.2215891803561598,
    0.012088191269830968,
    0.0018035955282857206,
    0.0005700909201064419,
    0.000178308050988675,
    8.682284674234989e-05,
    6.078516676220591e-05,
    3.730942344956498e-05,
    1.797065785087754e-05,
    2.5332947334243387e-05,
    2.806603141375651e-05,
    3.801940742920128e-05,
    3.7576438328276034e-05,
    2.4433616634739063e-05,
    2.8634958794184913e-05,
    3.402696279147671e-05,
    2.1713188591550668e-05,
    4.2202950963342604e-05,
    1.8443692896370576e-05,
    2.610597489572449e-05,
]


CT_expected = [0.0022 ** (x) * 100 for x in range(0, 21)]


plt.figure(figsize=(10, 8))
plt.rcParams.update({"font.size": 22})
plt.title("Average cross-talk probability")
plt.xlabel("Distance in pixels [-]")
plt.ylabel("Cross-talk probability [%]")
plt.yscale("log")
plt.errorbar(
    on_both_average.keys(),
    [x[0] for x in on_both_average.values()],
    yerr=[x[1] for x in on_both_average.values()],
    fmt=".",
    color="darkred",
    label="Measured",
    markersize=12,
)
plt.plot(
    [x for x in range(0, 21)],
    CT_expected,
    color="teal",
    label="Exponential 0.22%**distance",
)

plt.tight_layout()
plt.ylim(1e-6, 120)
plt.legend()


############################

import os
from glob import glob

from LinoSPAD2.functions import cross_talk, plot_tmsp

path = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_#33"

os.chdir(path)
files = glob("*.dat")

# plot_tmsp.collect_data_and_apply_mask(
#     files,
#     daughterboard_number="NL11",
#     motherboard_number="#21",
#     firmware_version="2212s",
#     timestamps=1000,
#     include_offset=False,
#     app_mask=False,
#     save_to_file=True,
# )

hot_pixels = [
    15,
    50,
    52,
    66,
    93,
    98,
    109,
    122,
    210,
    231,
    236,
]

# cross_talk.zero_to_cross_talk_collect(
#     path,
#     hot_pixels,
#     rewrite=True,
#     daughterboard_number="NL11",
#     motherboard_number="#33",
#     firmware_version="2212s",
#     timestamps=1000,
#     include_offset=False,
# )

cross_talk.zero_to_cross_talk_plot(path, hot_pixels, show_plots=False)
