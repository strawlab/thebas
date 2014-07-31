# coding=utf-8
"""Code for sinewave-perturbation-response bayesian (and others) data analysis."""
import os.path as op


def matplotlib_without_x(force=False):
    import os
    if force or os.getenv('DISPLAY') is None:
        import matplotlib
        matplotlib.use('agg')
        from matplotlib import pyplot as plt
        plt.ioff()
matplotlib_without_x()

# One dir to contain'em all
PERTURBATION_BIAS_ROOT = op.join(op.expanduser('~'), 'data-analysis', 'closed_loop_perturbations')
if not op.isdir(PERTURBATION_BIAS_ROOT):
    PERTURBATION_BIAS_ROOT = '/mnt/strawscience/santi/dcn-tethered-bayesian'
# Where the original (and munged) data will be
PERTURBATION_BIAS_DATA_ROOT = op.join(PERTURBATION_BIAS_ROOT, 'data')
PERTURBATION_BIAS_SILENCED_FLIES = op.join(PERTURBATION_BIAS_DATA_ROOT, 'VT37804_TNTE')
PERTURBATION_BIAS_KINDAWT_FLIES = op.join(PERTURBATION_BIAS_DATA_ROOT, 'VT37804_TNTin')
# An HDF5 file to test
TEST_HDF5 = op.join(PERTURBATION_BIAS_DATA_ROOT, 'VT37804_TNTE', '2012-12-18-16-04-06.hdf5')
# Where we will store MCMC traces
DEFAULT_MCMC_RESULTS_DIR = op.join(PERTURBATION_BIAS_ROOT, 'MCMC')
# Where some of the generated plots will be
DEFAULT_PLOTS_DIR = op.join(DEFAULT_MCMC_RESULTS_DIR, 'plots')