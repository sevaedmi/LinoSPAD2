import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

x1 = np.array(
    [
        100.00,
        89.94,
        75.88,
        60.30,
        50.24,
    ]
)
x2 = np.array(
    [
        100.00,
        90.16,
        71.83,
        61.98,
        46.47,
    ]
)

hbt1 = np.array([203, 177, 95, 36, 19])
ct1 = np.array([292, 293, 230, 203, 164])
hbt2 = np.array([292, 267, 101, 67, 39])
ct2 = np.array([404, 314, 283, 212, 208])


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, c):
    return a * x**2 + c


pars_ct1, covs_ct1 = curve_fit(linear, x1, ct1)
pars_hbt1, covs_hbt1 = curve_fit(quadratic, x1, hbt1)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x1, ct1)
ax.plot(x1, linear(x1, *pars_ct1))
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x1, hbt1)
ax.plot(x1, quadratic(x1, *pars_hbt1))
plt.show()
