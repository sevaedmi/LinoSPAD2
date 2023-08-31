# Introduction

This package was written for data analysis for LinoSPAD2, mainly for
analysis of the timestamp output. The key functions are ones for
unpacking the binary output of the detector that utilizes the numpy
library for quick unpacking of .dat files to matrices,
dictionaries, or data frames.

The "functions" folder holds all functions from unpacking to plotting
numerous types of graphs (pixel population, histograms of timestamp
differences, etc.)

The "params" folder holds masks (used to mask the noisiest pixels) and
calibration data (compensating for TDC nonlinearities and offset) for
LinoSPAD2 daughterboards "A5" and "NL11".

The "archive" folder is a collection of scripts for debugging, tests,
older versions of functions, etc.

Some functions (mainly the plotting ones) save plots as pictures in the
.png format, creating a folder for the output in the same folder that
holds the data. Others (such as delta_t.py for collecting timestamp differences
in the given time window) save .csv files with the processed data for
easier and faster plotting.

Additionally, a standalone repo with an application for online plotting
of the sensor population can be found [here](https://github.com/rngKomorebi/LinoSPAD2-app).
