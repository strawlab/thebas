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

# Where we will store MCMC traces
DEFAULT_MCMC_RESULTS_DIR = op.join(op.abspath(op.dirname(__file__)), 'MCMC')
# Plots
DEFAULT_PLOTS_DIR = op.join(DEFAULT_MCMC_RESULTS_DIR, 'plots')