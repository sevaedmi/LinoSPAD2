# Installation and dependencies

To start using the module, one can download the whole repo. The 'main.py'
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
