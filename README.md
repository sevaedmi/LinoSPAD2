# Introduction
This module was written for data analysis for LinoSPAD2, mainly for
analysis of the timestamp output. The key functions are ones for
unpacking the binary output of the detector that utilize the numpy
Python library for quick unpacking of .dat files to matrices,
dictionaries or dataframes.

The "functions" folder holds all functions from unpacking to plotting
numerous types of graphs (pixel population, histograms of timestamp
differences, etc.)

The "params" folder holds masks (used to mask some of the noisiest
pixels) and calibration data (compensating for TDC nonlinearities and
offset) for LinoSPAD2 daughterboards "A5" and "NL11".

The "archive" folder is a collection of scripts for debugging, tests,
older versions of functions, etc.

A full documentation, including examples and a full documentation of
modules and functions, can be found at TODO:insert link to doc.

Additionaly, a standalone repo with an application for online plotting
of sensor population can be found [here](https://github.com/rngKomorebi/LinoSPAD2-app).

Some functions (mainly the plotting ones) save plots as pictures in the
.png format, creating a folder for the output in the same folder that
holds the data.

# Installation and usage

To start using the module, one can download the whole repo. The 'main.py'
serves as the main hub for calling the functions. "requirements.txt"
collects all packages required for this project to run. One can create
an environment for this project either using conda or install the
necessary packages using pip.
```
pip install -r requirements.txt
```
or
```
conda create --name NEW_ENVIRONMENT_NAME --file /PATH/TO/requirements.txt -c conda-forge
```

# How to contribute
This repo consists of two branches: 'main' serves as the release version
of the module, tested, proved to be functional and ready-to-use, while
'develop' branch serves as the main hub for testing new stuff. To
contribute, the best way would be to fork the 'develop' branch and
submit via pull requests. Everyone willing to contribute is kindly asked
to follow the [PEP 8](https://peps.python.org/pep-0008/) and
[PEP 257](https://peps.python.org/pep-0257/) in particular.
