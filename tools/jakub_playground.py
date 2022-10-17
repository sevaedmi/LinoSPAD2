import numpy as np
from functions import delta_t, plot_valid, fits as gf



def main_playground():
    path_Ne_lamp = r"C:\Users\jakub\Documents\LinoSpad2\data\SPDC_221014\Analyze"
    delta_t.plot_grid(path_Ne_lamp, (203,204, 205,206,207), show_fig=False, same_y=True)
