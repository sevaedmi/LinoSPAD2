# Installation and dependencies

To start using the package, one can download the whole repo. The 'main.py'
serves as the main hub for calling the functions. "requirements.txt"
collects all packages required for this project to run. One can create
an environment for this project either using conda or pip (for creating
virtual environments using pip
see [this](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)):

Using pip:
```
pip instal virtualenv
py -m venv PATH/TO/NEW/ENVIRONMENT/NEW_ENVIRONMENT_NAME
PATH/TO/NEW/ENVIRONMENT/NEW_ENVIRONMENT_NAME/Scripts/activate
cd PATH/TO/THIS/PACKAGE
pip install -r requirements.txt
pip install -e .
```
where the last command installs the package itself.

Using conda:
```
cd PATH/TO/THIS/PACKAGE
conda create --name NEW_ENVIRONMENT_NAME --file requirements.txt -c conda-forge
conda activate NEW_ENVIRONMENT_NAME
conda develop .
```
Alternatively, if one wishes to implement the package in the already
existing environment, skipping the creation of a new environment and
installation of the package is advised. In any case, after the package
is installed, one can import the functions:
```
from LinoSPAD2.functions import plot_tmsp, delta_t, fits
```