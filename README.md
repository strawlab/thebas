Bayesian Analysis of Strawlab Tethered Data
-------------------------------------------

All dependencies are either declared in *setup.py* or hardcopied in the *externals* package.
 
A burden-free way to run this code is to use strz.
There all the dependencies should already installed when using anaconda python in */opt/anaconda*.

```sh
# All dependencies should already be installed in strz 
# (at least in the anaconda distro in /opt)
ssh strawlab@strz
# Clone the repository
git clone https://github.com/strawlab/thebas
```

For the perturbation analysis experiments, it is easy to reproduce our results.
 
```sh
# Go to the cloned repository
cd thebas

# The data is read from 
#   *"/mnt/strawscience/data/forSANTI/closed_loop_perturbations"*.
# This munges these data into a more suitable format for analysis 
# It is an example of data loading that also does some crazy printing
PYTHONPATH=$PYTHONPATH:. /opt/anaconda/bin/python thebas/sinefitting/perturbation_experiment.py

# This shows examples of command lines that can be used to fit bayesian models...
PYTHONPATH=$PYTHONPATH:. /opt/anaconda/bin/python thebas/sinefitting/samplers.py cl

# To run many fittings in parallel, I use the parallel command
PYTHONPATH=$PYTHONPATH:. /opt/anaconda/bin/python thebas/sinefitting/samplers.py cl &>commands
cat commands | parallel -j12

# Let's toy-fit one model, 4 chains, only 800 iterations...
PYTHONPATH=$PYTHONPATH:. /opt/anaconda/bin/python -u thebas/sinefitting/samplers.py sample --freq 40 --genotype-id VT37804_TNTE --mol-id gpa3nomap --iters 800 --burn 400 &>~/gpa3nomap__VT37804_TNTE__40.log
# To add new models, just edit thebas/sinefitting/models (warning loads of copy and paste there!)
# Copy a model to modify and add it at the end to the dictionary "MODEL_FACTORIES"

# This has stored the traces and generated plots here:
ls thebas/sinefitting/MCMC/

# There is a bunch or dirty results-analysis code in thebas/sinefitting/reports 
# (warning, really awful and incomplete code lives there).
```