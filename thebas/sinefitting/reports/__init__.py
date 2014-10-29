# coding=utf-8
from itertools import product
import os.path as op
from glob import glob
import re

from pandas import DataFrame

from thebas.sinefitting.results import MCMCRunManager


# ---- Consistency between plots and other cosmetics


def mpl_params():
    # http://nbviewer.ipython.org/github/CamDavidsonPilon/
    # Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/IntroMCMC.ipynb
    # http://stackoverflow.com/questions/15814635/prettier-default-plot-colors-in-matplotlib
    from matplotlib import rc, rcParams
    rc('font', family='serif', size=16)
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\boldmath']
    try:
        import seaborn  # seaborn defaults are nicer
    except:
        pass


def genotype_color(genotype):
    return 'b' if 'tntin' in genotype.lower() else 'r'


# ---- Results I/O


def result_coords_from_path(path):
    """ Infer the experiment coordinates (model_id, freq and genotype) from a directory name.

    Example:
    >>> model_id, freq, genotype = result_coords_from_path('model=gp1__freq=0.5__genotype=VT37804_TNTE')
    >>> model_id
    'gp1'
    >>> freq
    0.5
    >>> genotype
    'VT37804_TNTE'
    """
    model_id, freq, genotype = re.search('model=(.*)__freq=(.*)__genotype=(.*)', path).groups()
    return model_id, float(freq), genotype


def all_computed_results(results_dir):
    """Returns all the results under a directory as a pandas dataframe.
    The columns are: 'model_id', 'freq', 'genotype', 'path'
    """
    results_dirs = filter(op.isdir, sorted(glob(op.join(results_dir, 'model=*'))))
    return DataFrame(map(lambda r: list(result_coords_from_path(r)) + [r], results_dirs),
                     columns=('model_id', 'freq', 'genotype', 'path'))


def flies_and_variables(result, refvar='phase'):
    """
    Really ad-hoc
    We assume a model in which...
      - there is one hyperfly
      - there are many flies (all with the same parameterization)
    """
    trace_names = result.varnames()[0]  # Assume ATM that the first chain has all the variables of interest
    hyperfly = result.name.partition('__')[2].rpartition('__')[0]  # Nasty
    hyperfly_postfix = '_group="%s"' % hyperfly
    hyperfly_variables = [tn[0:-len(hyperfly_postfix)] for tn in trace_names if tn.endswith(hyperfly_postfix)]
    flies = sorted(set(tn[len('%s_' % refvar):] for tn in trace_names
                       if tn.startswith('%s_' % refvar)) - {hyperfly})
    flies_variables = [tn[0:-len('_' + flies[0])] for tn in trace_names if tn.endswith('_' + flies[0])]
    return hyperfly, hyperfly_postfix, hyperfly_variables, flies, flies_variables


def cache_all_traces(pbproject):
    """Cache all traces to allow quick retrieval, beyond pymc pickledb madness."""
    for result_path in all_computed_results(pbproject.mcmc_dir).path:
        result = MCMCRunManager(result_path)
        hyperfly, hyperfly_postfix, hyperfly_variables, flies, flies_variables = flies_and_variables(result)
        result.num_chains()
        # Cache all traces for quick retrieval
        for var in hyperfly_variables:
            result.traces(var + '_' + hyperfly)
        for fly, var in product(flies, flies_variables):
            result.traces(var + '_' + fly)

#
# other possible plots:
# - Box-plot of individual flies
# - Autocorrelation of group estimates
# - Real data over fitted curve and vs ideal curve
# - DC vs Amplitude
#
