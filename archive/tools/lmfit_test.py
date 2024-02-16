import numpy as np
import pandas as pd
from lmfit.models import ConstantModel, GaussianModel
from matplotlib import pyplot as plt
from pyarrow import feather as ft

# file = "D:/LinoSPAD2/Data/board_NL11/BNL/FW_2212b/Ne_703/delta_ts_data/0000050776-0000051075.csv"
# file = "D:/LinoSPAD2/Data/board_A5/BNL/FW_2212_block/SPDC/delta_ts_data/0000000050-0000000089.csv"

file = r"D:\LinoSPAD2\Data\board_NL11\Prague\CT_HBT\Second try\CT_HBT_1-0m_80%\delta_ts_data\0000337767-0000338066.feather"

# data = pd.read_csv(file)
data = ft.read_feather(
    "{}".format(file),
    columns=[f"{170},{174}"],
).dropna()

x = np.arange(256)

y = np.array(data.values.T)[0]
print(data)
print(y)

to_del = np.where(np.logical_or(y < -10e3, y > 10e3))[0]

y1 = np.delete(y, to_del)

step = 2

bins = np.arange(np.min(y1), np.max(y1), 17.857 * step)

n, b, p = plt.hist(y1, bins=bins)
plt.close()

b_c = (b - (b[-1] - b[-2]) / 2)[1:]

mod_peak = GaussianModel()
mod_bckg = ConstantModel()
mod_res = mod_peak + mod_bckg

peak_pos = np.where(n == np.max(n))[0][0].astype(int)

y_bckg = np.copy(n)
y_bckg[peak_pos - 10 : peak_pos + 10] = np.median(n)


pars = mod_peak.guess(y_bckg, x=b_c)
pars += mod_bckg.guess(y_bckg, x=b_c)

res = mod_res.fit(n, pars, x=b_c)

print(res.fit_report(min_correl=0.25))

amp_fit = res.params["amplitude"].value
mean_fit = res.params["center"].value
sigma_fit = res.params["sigma"].value
background_fit = res.params["c"].value


amp_stderr = res.params["amplitude"].stderr
mean_stderr = res.params["center"].stderr
sigma_stderr = res.params["sigma"].stderr
background_stderr = res.params["c"].stderr

print(amp_stderr, mean_stderr, sigma_stderr, background_stderr)


plt.figure(figsize=(16, 10))
res.plot_fit(title="Gumba", fit_kws={"color": "salmon"})
dely = res.eval_uncertainty(sigma=3)
plt.fill_between(
    b_c,
    res.best_fit - dely,
    res.best_fit + dely,
    color="#ABABAB",
    label="3-\u03C3 uncertainty band",
)
plt.legend()
