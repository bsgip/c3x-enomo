# c3x-enomo (ENergy Output Model Optimiser)

To set environment variable for location of optimiser (cplex etc).
https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux

Locate the directory for the conda environment in your terminal window by running in the terminal echo $CONDA_PREFIX.

Enter that directory and create these subdirectories and files:
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

Edit ./etc/conda/activate.d/env_vars.sh as follows:
	#!/bin/sh
	export OPTIMISER_ENGINE='cplex'
	export OPTIMISER_ENGINE_EXECUTABLE=/opt/ibm/ILOG/CPLEX_Studio128/cplex/bin/x86-64_linux/cplex

Edit ./etc/conda/deactivate.d/env_vars.sh as follows:
	#!/bin/sh
	unset OPTIMISER_ENGINE
	unset OPTIMISER_ENGINE_EXECUTABLE

When you run conda activate analytics, the environment variables MY_KEY and MY_FILE are set to the values you wrote into the file. When you run conda deactivate, those variables are erased.