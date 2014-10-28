# coding=utf-8
"""Bayesian BODE plots."""
from user import home
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
from pymc import gelman_rubin
from pymc.utils import hpd

from thebas.sinefitting.reports.reports_jumble import flies_and_variables, all_computed_results
from thebas.sinefitting.results import MCMCRunManager


def bodelike_plot(model_id='gpa3',
                  varname='phase',
                  num_chains=4, takelast=10000,
                  alpha=0.05,
                  plot_control=True, plot_silenced=True, img_format='png',
                  show=False):

    def varnames(result, varname):
        hyperfly, hyperfly_postfix, hyperfly_variables, flies, flies_variables = flies_and_variables(result)
        hvar = varname + hyperfly_postfix if varname in set(hyperfly_variables) else None
        fvars = [varname + '_' + fly for fly in flies] if varname in set(flies_variables) else None
        return hvar, fvars

    def mix_chains(chains):
        # assert len(chains) >= num_chains
        mixed = np.array([np.nan] * (num_chains * takelast))
        for i, chain in enumerate(chains):
            mixed[i * takelast: (i+1) * takelast] = chain[-takelast:]
        return mixed

    # Available results
    results = all_computed_results()
    results = results[results.model_id == model_id]
    ctraces = {}
    straces = {}
    # FIXME: here we got wrong genotypes...
    results.genotype = results.genotype.apply(lambda gen: gen.partition('__')[0])
    # Collect and mix traces for all frequencies
    for (model_id, freq), data in results.groupby(('model_id', 'freq')):
        print '\t\t\tCollecting traces for frequency %g' % freq
        control = MCMCRunManager(data[data.genotype == 'VT37804_TNTin'].iloc[0]['path'])  # ad-hoc
        silenced = MCMCRunManager(data[data.genotype == 'VT37804_TNTE'].iloc[0]['path'])  # ad-hoc
        chvar, _ = varnames(control, varname)   # control hierarchical var, fly vars
        shvar, _ = varnames(silenced, varname)  # silenced hierarchical var, fly vars
        ctraces[freq] = mix_chains(control.traces(chvar))
        straces[freq] = mix_chains(silenced.traces(shvar))
    # The frequencies we are interested in...
    freqs = (0.5, 1, 2, 4, 8, 16, 32, 40)
    # Copute HPDs. Compute the rope too, see Kruschke.
    chpds = [hpd(ctraces[freq], alpha) for freq in freqs]
    shpds = [hpd(straces[freq], alpha) for freq in freqs]
    # Plot the traces
    if plot_control:
        plt.plot(np.hstack([ctraces[freq] for freq in freqs]), color='b', label='control')
    if plot_silenced:
        plt.plot(np.hstack([straces[freq] for freq in freqs]), color='r', label='silenced')
    # Plot the HPD regions + setup ticks
    xticklocations = []
    xticklabels = []
    for i, freq in enumerate(freqs):
        xmin = num_chains * takelast * i
        xmax = num_chains * takelast * (i + 1)
        plt.axvline(x=xmax, color='k')
        plt.plot((xmin, xmax), [chpds[i][0]] * 2, color='c', linewidth=4)
        plt.plot((xmin, xmax), [chpds[i][1]] * 2, color='c', linewidth=4)
        plt.plot((xmin, xmax), [shpds[i][0]] * 2, color='m', linewidth=4)
        plt.plot((xmin, xmax), [shpds[i][1]] * 2, color='m', linewidth=4)
        # Gelman-Rubin R^2 (might interest: Geweke, autocorr, put graphically in the plot)
        cgr = gelman_rubin(ctraces[freq].reshape(num_chains, -1))
        print '\t%s %s control freq %.1f; GR=%.2f' % (model_id, varname, freq, cgr)
        sgr = gelman_rubin(straces[freq].reshape(num_chains, -1))
        print '\t%s %s silence freq %.1f; GR=%.2f' % (model_id, varname, freq, sgr)
        # xticks
        xticklocations.append(xmin + (xmax - xmin) / 2.)
        xticklabels.append('%g\nsgr=%.2f\ncgr=%.2f' % (freq, sgr, cgr))
    plt.title('Model: %s; Variable: %s' % (model_id, varname))
    plt.xlabel('$\omega$')
    plt.ylabel('%s' % varname)
    plt.tick_params(axis='x',           # changes apply to the x-axis
                    which='both',       # both major and minor ticks are affected
                    top='off',          # ticks along the top edge are off
                    bottom='on',        # ticks along the bottom edge are on
                    labelbottom='on')   # labels along the bottom edge are off
    plt.xticks(xticklocations, xticklabels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(op.join(home, '%s-%s.%s' % (model_id, varname, img_format)))
    if show:
        plt.show()


def failsafe_plot(model, var, fignum=0, show=False):
    if var == 'amplitude' and 'gpa' not in model:
        return
    print model, var
    try:
        plt.figure(fignum)
        bodelike_plot(model_id=model, varname=var, show=show)
        plt.close(fignum)
    except Exception, e:
        print '\tFailed: %s' % str(e)


if __name__ == '__main__':

    models_hypervars = (
        ('gpa_t1', ('amplitudeAlpha', 'amplitudeBeta', 'phaseKappa', 'phaseMu')),
        ('gpa_t1_slice', ('amplitudeAlpha', 'amplitudeBeta', 'phaseKappa', 'phaseMu')),
        ('gpa_t2_slice', ('amplitudeAlpha', 'amplitudeBeta', 'phaseKappa', 'phaseMu')),
        ('gpa3', ('amplitude', 'phase')),
        ('gpa3hc1', ('amplitude', 'phase')),
        ('gpa3hc2', ('amplitude', 'phase')),
    )

    fignum = 0
    for model, variables in models_hypervars:
        for variable in variables:
            print 'model=%s, variable=%s' % (model, variable)
            failsafe_plot(model, variable, fignum=fignum, show=False)
            fignum += 1