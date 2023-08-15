Scripts for unpacking and analyzing data collected with LinoSPAD2. The data
should be binary-encoded (the ".dat" format) as the ".txt" format takes much longer
to unpack. The 1D binary streams of data from the detector can be unpacked
and saved with the "unpack" module in the "functions" directory.

"main.py" is the main hub where individual modules are called. Modules for
real time plotting of the number of valid timestamps vs the pixel index
("plot_valid"), for plotting a grid of differences in timestamps for a given
range of pixels ("delta_t"), and for fitting the timestamp differences with
a gaussian function are available ("fits").

The "tools" directory contains numerous scripts, some of which mirror some of
the modules in the "functions" as a standalone scripts that can be used as a
debugging tool. Other scripts were used in the testing of new functions.

'_old' directories contain previous iterations of the scripts that are left for
later reference, debugging, or insipiration.

The 'app' directory contains scripts that run an application for data analysis,
namely realtime plotting of timestamp counts in each pixel for better focusing
during manipulations with the setup. However, the app was moved to a standalone
directory, https://github.com/rngKomorebi/LinoSPAD2-app and the version in this
repo is outdated.

Folders 'masks' and 'calibration_data' serve as parameter holders for some of
the functions, containing masks for hot/warm pixels and TDC nonlinearity calibration
data for LinoSPAD2 NL11 and A5 daughterboards, respectively.

'jupyter' folder collects functions written in Jupyter by some of the contributors.

"requirements.txt" collects all packages required for this project to run.
One can create an environment for this project either using conda or  install
necessary packages using pip.
```
pip install -r requirements.txt
or
```
```
conda create --name NEW_ENVIRONMENT_NAME --file /PATH/TO/requirements.txt -c conda-forge
```
