import numpy as np
from functions import delta_t, plot_valid, fits as gf



def main_playground():
    path_Ne_lamp = r"C:\Users\jj\Documents\LinoSpad\First_attemps"
    delta_t.plot_grid(path_Ne_lamp, (156,157, 158, 159, 160), show_fig=False, same_y=True)
