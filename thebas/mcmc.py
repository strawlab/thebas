# coding=utf-8

import pymc
from pymc.database.base import batchsd

import numpy as np
from pymc.utils import hpd


# ---- MCMC Chain Stats

def standardized_autocorrelation(x):
    # from http://tinyurl.com/afz57c4
    #  and http://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
    acorr = np.correlate(x, x, mode='full')
    acorr = acorr / np.max(acorr)
    return acorr[acorr.size / 2:]


def trace_stats(trace, alpha=0.05, batches=100, quantiles=(2.5, 25, 50, 75, 97.5)):
    """
    Generate posterior statistics for the trace ala pymc (this was adapted from pymc.database.base)

    :Parameters:

    trace : ndarray

    alpha : float
      The alpha level for generating posterior intervals. Defaults to
      0.05.

    start : int
      The starting index from which to summarize (each) chain. Defaults
      to zero.

    batches : int
      Batch size for calculating standard deviation for non-independent
      samples. Defaults to 100.

    chain : int
      The index for which chain to summarize. Defaults to None (all
      chains).

    quantiles : tuple or list
      The desired quantiles to be calculated. Defaults to (2.5, 25, 50, 75, 97.5).
    """

    trace = np.squeeze(trace)

    return {
        'n': len(trace),
        'standard deviation': trace.std(0),
        'mean': trace.mean(0),
        '%s%s HPD interval' % (int(100 * (1 - alpha)), '%'): hpd(trace, alpha),
        'mc error': batchsd(trace, batches),
        'quantiles': pymc.utils.quantiles(trace, qlist=quantiles),
        'acorr': standardized_autocorrelation(trace)  # [1:]
    }
