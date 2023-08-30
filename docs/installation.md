# Installation and dependencies

To start using the package, one can download the whole repo. The 'main.py'
serves as the main hub for calling the functions. "requirements.txt"
collects all packages required for this project to run. One can create
an environment for this project either using conda or install the
necessary packages using pip (for creating virtual envonments using pip
see [this](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)):
```
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
and run
```
pip install -e .
```
that will install the local package LinoSPAD2. After that, you can
import all functions in your project:
```
from LinoSPAD2.functions import plot_tmsp, delta_t, fits
```