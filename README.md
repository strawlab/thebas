Bayesian Analysis of Strawlab Tethered Data
===========================================

[Find discussions in the wiki](https://github.com/strawlab/thebas/wiki).

Install
-------

All dependencies are either declared in *setup.py* or hardcopied in the *externals* package. This makes
everything work on current anaconda (as per 2015/03/23).

- Install [conda](http://www.conda.io) / [anaconda](http://continuum.io/downloads).
- Install thebas dependencies:
```sh
conda install h5py pymc numpy scipy pandas matplotlib dill
conda install pydot  # this will unfortunatelly downgrade pyparsing
pip install joblib argh
```
- Install thebas. Either clone the repository and install in development mode (recommended)...
 
```sh
# Clone the repository
git clone https://github.com/strawlab/thebas
pip install -e thebas
```

... or pip install-it from github

```sh
pip install 'git+git://github.com/strawlab/thebas.git'
```

Everything should already be installed in strz (using the default anaconda environment in */opt/anaconda*).

Use
---

This is a small tour on the different scripts provided.
 
```sh
# Activate anaconda or just use anaconda's python executable
alias python='/opt/anaconda/bin/python'

# Go to the cloned repository
cd thebas

# The data is read from 
#   *"/mnt/strawscience/data/forSANTI/closed_loop_perturbations"*.
# This munges these data into a more suitable format for analysis 
# This is an example of data loading that also does some crazy printing
python thebas/sinefitting/perturbation_experiment.py

# This shows examples of command lines that can be used to fit bayesian models...
python thebas/sinefitting/samplers.py cl

# To run many fittings in parallel, we can use gnu parallel
python thebas/sinefitting/samplers.py cl | parallel -j12

# Let's toy-fit one model, 4 chains, only 800 iterations...
python -u thebas/sinefitting/samplers.py sample --pbproject dcn --freq 4 --genotype-id VT37804_TNTE --model-id gpa_t3 --iters 800 --burn 400 &>~/dcn__gpa_t3__VT37804_TNTE__4.log
# To add new models, just edit thebas/sinefitting/models (warning loads of useful copy and paste there!)
# Copy a model to modify and add it at the end to the dictionary "MODEL_FACTORIES"
# pbproject can be one of "dcn" or "hs"

# The traces and generated plots stay here:
ls /mnt/strawscience/santi/dcn-tethered

# --- There are some results-analysis scripts in thebas/sinefitting/reports

# Plot bias against the wing-beat-amplitude
python -u thebas/sinefitting/reports/bias_vs_wba.py

# Generate the "Bayesian BODE plots"
python -u thebas/sinefitting/reports/bbode.py

# Run NHST wannabes
python -u thebas/sinefitting/reports/nhst.py

# Show information about the stimulus perturbation
python -u thebas/sinefitting/reports/perturbation.py 
```
