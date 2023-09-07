# LinoSPAD2

Package for unpacking and analyzing the binary data from LinoSPAD2.

## Introduction

This package was written for data analysis for LinoSPAD2, mainly for
analysis of the timestamp output. The key functions are ones for
unpacking the binary output of the detector that utilizes the numpy
Python library for quick unpacking of .dat files to matrices,
dictionaries, or data frames.

The "functions" folder holds all functions from unpacking to plotting
numerous types of graphs (pixel population, histograms of timestamp
differences, etc.)

The "params" folder holds masks (used to mask some of the noisiest
pixels) and calibration data (compensating for TDC nonlinearities and
offset) for LinoSPAD2 daughterboards "A5" and "NL11".

The "archive" folder is a collection of scripts for debugging, tests,
older versions of functions, etc.

Full documentation, including examples and full documentation of
modules and functions, can be found [here](https://rngkomorebi.github.io/LinoSPAD2/).

Some functions (mainly the plotting ones) save plots as pictures in the
.png format, creating a folder for the output in the same folder that
holds the data. Others (such as delta_t.py for collecting timestamp differences
in the given time window) save .csv files with the processed data for
easier and faster plotting.

Additionally, a standalone repo with an application for online plotting
of the sensor population can be found [here](https://github.com/rngKomorebi/LinoSPAD2-app).

## Installation and usage

To start using the package, one can download the whole repo. The 'main.py'
serves as the main hub for calling the functions. "requirements.txt"
collects all packages required for this project to run. One can create
an environment for this project either using conda or install the
necessary packages using pip (for creating virtual environments using pip
see [this](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)):
```
pip install virtualenv
py -m venv PATH/TO/ENVIRONMENT/ENVIRONMENT_NAME
PATH/TO/ENVIRONMENT/ENVIRONMENT_NAME/Scripts/activate
pip install -r requirements.txt
```
or (recommended)
```
conda create --name NEW_ENVIRONMENT_NAME --file /PATH/TO/requirements.txt -c conda-forge
```
To install the package, first, switch to the created environment:
```
conda activate NEW_ENVIRONMENT_NAME
```
or
```
PATH/TO/ENVIRONMENT/ENVIRONMENT_NAME/Scripts/activate
```
and run
```
pip install -e .
```
that will install the local package LinoSPAD2. After that, you can
import all functions in your project:
```
from LinoSPAD2.functions import plot_tmsp, delta_t, fits
```

## How to contribute

This repo consists of two branches: 'main' serves as the release version
of the package, tested, proven to be functional, and ready to use, while
the 'develop' branch serves as the main hub for testing new stuff. To
contribute, the best way would be to fork the 'develop' branch and
submit via pull requests. Everyone willing to contribute is kindly asked
to follow the [PEP 8](https://peps.python.org/pep-0008/) and
[PEP 257](https://peps.python.org/pep-0257/) conventions.

## License and contact info

This package is available under the MIT license. See LICENSE for more
information. If you'd like to contact me, the author, feel free to
write at sergei.kulkov23@gmail.com.
