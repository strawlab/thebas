# coding=utf-8
"""Building reports out of the initial data and the MCMC results for the sinewave perturbation data experiment."""
from itertools import product
import os.path as op
from glob import glob
from user import home
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import pymc
from pymc.database.base import batchsd
from pymc.utils import hpd
from pymc.diagnostics import gelman_rubin
from pandas import DataFrame
import matplotlib.pyplot as plt
from thebas.externals.tethered_data.examples.perturbation_experiment import perturbation_data_to_records, all_biases
from thebas.misc import ensure_dir
from thebas.sinefitting import DEFAULT_MCMC_RESULTS_DIR, DEFAULT_PLOTS_DIR
from thebas.sinefitting.models import perturbation_signal
from thebas.sinefitting.results import MCMCRunManager


MCMC_TEST_RESULT = op.join(DEFAULT_MCMC_RESULTS_DIR, 'model=gp1__freq=0.5__genotype=VT37804_TNTE')


###########################################
# CONSISTENCY BETWEEN PLOTS and other cosmetics
###########################################

def mpl_params():
    # http://nbviewer.ipython.org/github/CamDavidsonPilon/
    # Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter3_MCMC/IntroMCMC.ipynb
    # http://stackoverflow.com/questions/15814635/prettier-default-plot-colors-in-matplotlib
    from matplotlib import rc, rcParams
    rc('font', family='serif', size=16)
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\boldmath']
    try:
        from mpltools import style
        style.use('ggplot')
    except:
        pass


def genotype_color(genotype):
    return 'b' if genotype.endswith('in') else 'r'


def detect_discontinuities(y, threshold):
    return np.where(np.abs(np.diff(y)) >= threshold)[0]


def mark_discontinuities(x, y, threshold=0.01, use_x=False):
    discontinuities = detect_discontinuities(y if not use_x else x, threshold) + 1
    x = np.insert(x, discontinuities, np.nan)
    y = np.insert(y, discontinuities, np.nan)
    return x, y


###########################################
# DATA I/O
###########################################


def result_coords_from_path(path=MCMC_TEST_RESULT):
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


def all_computed_results(results_dir=DEFAULT_MCMC_RESULTS_DIR):
    """Returns all the results under a directory as a pandas dataframe.
    The columns are: 'model_id', 'freq', 'genotype', 'path'
    """
    results_dirs = filter(op.isdir, sorted(glob(op.join(results_dir, 'model=*'))))
    return DataFrame(map(lambda r: list(result_coords_from_path(r)) + [r], results_dirs),
                     columns=('model_id', 'freq', 'genotype', 'path'))


def flies_and_variables(result, refvar='phase', varname_first=True):
    """
    Really ad-hoc
    We assume a model in which...
      - One hyperfly
      - Many flies (all with the same parameterization)
    """
    trace_names = result.varnames()[0]  # Assume ATM that the first chain has all the variables of interest
    hyperfly = result.name.partition('__')[2]  # Nasty
    if varname_first:
        hyperfly_variables = [tn[0:-len('_' + hyperfly)] for tn in trace_names if tn.endswith('_' + hyperfly)]
        flies = sorted(set(tn[len('%s_' % refvar):] for tn in trace_names
                           if tn.startswith('%s_' % refvar)) - {hyperfly})
        flies_variables = [tn[0:-len('_' + flies[0])] for tn in trace_names if tn.endswith('_' + flies[0])]
    else:
        hyperfly_variables = [tn[len(hyperfly + '_'):] for tn in trace_names if tn.startswith(hyperfly + '_')]
        flies = sorted(set(tn[:-len('_%s' % refvar)] for tn in trace_names
                           if tn.endswith('_%s' % refvar)) - {hyperfly})
        flies_variables = [tn[len(flies[0] + '_'):] for tn in trace_names if tn.startswith(flies[0] + '_')]
    return hyperfly, hyperfly_variables, flies, flies_variables


def cache_all_traces():
    """Cache all traces to allow quick retrieval, beyond pymc pickledb madness."""
    for result_path in all_computed_results().path:
        result = MCMCRunManager(result_path)
        hyperfly, hyperfly_variables, flies, flies_variables = flies_and_variables(result)
        result.num_chains()
        # Cache all traces for quick retrieval
        for var in hyperfly_variables:
            result.traces(var + '_' + hyperfly)
        for fly, var in product(flies, flies_variables):
            result.traces(var + '_' + fly)


###########################################
# STATS
###########################################

def convergence_stats(traces):
    # See pymc.diagnostics
    from pymc import gelman_rubin

    gelman_rubin()


def autocorrelation(x):
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
        'acorr': autocorrelation(trace)  # [1:]
    }


###########################################
# PLOTS
###########################################


def all_biases_plot(transpose=True):
    # Original bias had amplitude 5
    from matplotlib import rc, rcParams
    rc('font', family='serif', size=16)
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\boldmath']
    bias, bias_t = all_biases().values()[0]
    bias_t -= bias_t[0]
    if not transpose:
        plt.plot(bias_t, bias, color='g')
        plt.ylabel('bias (rad/s)', fontsize=20)
        plt.xlabel('time (s)', fontsize=20)
        plt.xlim((-5, 205))
        plt.axhline(linewidth=4, color='g')
    else:
        plt.plot(bias, bias_t, color='g')
        plt.xlabel('bias (rad/s)', fontsize=20)
        plt.ylabel('time (s)', fontsize=20)
        plt.ylim((-5, 205))
        plt.axvline(linewidth=4, color='g')
    plt.title('bias velocity over time')
    plt.show()
    # Plot the bias magnitude of change in time...
    # dbias_t = bias_t[0:len(bias_t)-1] - bias_t[0]
    # dbias = bias[1:] - bias[0:len(bias)-1]
    dbias_t = bias_t
    dbias = np.gradient(bias)
    if not transpose:
        plt.plot(dbias_t, dbias, color='g')
        plt.axhline(linewidth=4, color='g')
        plt.ylabel('bias ($rad/s^2$)', fontsize=20)
        plt.xlabel('time (s)', fontsize=20)
        plt.xlim((-5, 205))
    else:
        plt.plot(dbias, dbias_t, color='g')
        plt.axvline(linewidth=4, color='g')
        plt.xlabel('bias ($rad/s^2$)', fontsize=20)
        plt.ylabel('time (s)', fontsize=20)
        plt.ylim((-5, 205))
    plt.title('bias acceleration over time')
    plt.show()


def data_sizes_plot(freqs=None):
    from matplotlib import rc, rcParams
    rc('font', family='serif', size=16)
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\boldmath']
    if freqs is None:
        freqs = np.array((0.5, 1, 2, 4, 8, 16, 32, 40))
    seconds = 2 * 8 * np.pi / freqs
    num_obs = seconds * 100
    width = .35
    ind = np.arange(len(freqs))
    plt.bar(ind, num_obs, width=0.35)
    plt.xticks(ind + width / 2, freqs)
    plt.ylabel('number of observations', fontsize=20)
    plt.xlabel('$\omega$ (rad/s)', fontsize=20)
    plt.title('Higher frequencies get less observations')
    plt.show()
    return num_obs


def bias_vs_wba_plots(img_format='png', transpose=True, legend=False):
    """Plot the bias and R-L together."""
    # TODO: zoom on high frequency to really see the range of change
    mpl_params()
    print('Reading the data...')
    df = perturbation_data_to_records()

    by_freq = df[['group', 'freq', 'fly', 'wba', 'wba_t']].groupby(('freq',))

    dest_dir = ensure_dir(op.join(DEFAULT_PLOTS_DIR, 'bias_vs_wba'))
    for freq, data in by_freq:
        min_t = min([wba_t.min() for wba_t in data['wba_t'] if len(wba_t) > 0])
        max_t = max([wba_t.max() for wba_t in data['wba_t'] if len(wba_t) > 0])
        min_wba = min([wba.min() for wba in data['wba'] if len(wba) > 0])
        max_wba = max([wba.max() for wba in data['wba'] if len(wba) > 0])
        for flyid, data in data.groupby(('fly',)):
            print(flyid, freq)
            if len(data) > 1:
                raise Exception('The flyid %s is not unique!' % flyid)
            # Fly wba trajectory
            data = data.iloc[0]
            group = data.group
            x, y = np.array(data['wba_t']), np.array(data['wba'])
            x, y = mark_discontinuities(x, y, use_x=True, threshold=0.02)
            # Ideal perturbation signal
            amplitude = (max_wba - min_wba) / 2.
            phase = 0
            mean_val = 0  # mean/median
            pert_t = np.linspace(min_t, max_t, 1000)
            pert = perturbation_signal(pert_t, amplitude, phase, mean_val, freq)
            # Superimpose both, plot
            plt.figure()
            if not transpose:
                plt.plot(x, y, color=genotype_color(data['group']), label='Fly R-L (rad)')
                plt.plot(pert_t, pert, color='g', label='Perturbation')
                plt.axhline(linewidth=4, color='g')
                plt.xlabel('time (s)')
                plt.ylabel('Strength of turn')
                if legend:
                    plt.legend()
                plt.xlim((min_t - (max_t - min_t) / 10, max_t + (max_t - min_t) / 10))
            else:
                plt.plot(y, x, color=genotype_color(data['group']), label='Fly R-L (rad)')
                plt.plot(pert, pert_t, color='g', label='Perturbation')
                plt.axvline(linewidth=4, color='g')
                plt.ylabel('time (s)')
                plt.xlabel('Strength of turn')
                if legend:
                    plt.legend()
                plt.ylim((min_t - (max_t - min_t) / 10, max_t + (max_t - min_t) / 10))
            plt.title('fly:%s(%s); freq=%.1f rad/s' %
                      (flyid, 'silenced' if group.endswith('E') else 'control', freq))
            fn = '%s-%.1f-%s.%s' % ('silenced' if group.endswith('E') else 'control', freq, flyid, img_format)
            plt.savefig(op.join(dest_dir, fn))
            plt.close()


def bias_vs_meanwba_plots(img_format='png', transpose=True, column_to_mean='wba', gradient=False):
    """Averages for each frequency and genotype and plots a wba vs bias signal plot."""
    #
    # TODO: When Lisa did this, how did she cope with different timings?
    # Will do it the naive but maybe incorrect way, just assuming we can nanmean all without regard for timing
    #
    # TODO: take into account that at higher frequencies amplitude must be much lower
    #
    from matplotlib import rc, rcParams
    rc('font', family='serif', size=16)
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\boldmath']

    def min_num_obs(group_data):
        return np.min([len(wba) for wba in group_data[column_to_mean]])  # Nasty

    def nanmean_xy(group_data, num_obs=None):
        if num_obs is None:
            num_obs = min_num_obs(group_data)
        wba = np.array([np.array(wba)[:num_obs] for wba in group_data[column_to_mean]])
        wba = np.nanmean(wba, axis=0)
        wba_t = np.array(group_data.irow(0)['wba_t'])[:num_obs]
        return wba_t, wba

    def one_group_plot(axes,
                       freq, group_name,
                       wba_t, wba,
                       min_t=None, max_t=None, min_wba=None, max_wba=None,
                       perturbation_amplitude=None):
        # gradient?
        if gradient:
            wba = np.gradient(wba)

        # Ranges
        min_t = np.min(wba_t) if min_t is None else min_t
        max_t = np.max(wba_t) if max_t is None else max_t
        min_wba = np.min(wba) if min_wba is None else min_wba
        max_wba = np.max(wba) if max_wba is None else max_wba

        # Ideal perturbation signal
        amplitude = (max_wba - min_wba) / 2. if perturbation_amplitude is None else perturbation_amplitude
        phase = 0
        mean_val = 0  # we center at zero to spot better flies that were turning always in the same direction
        pert_t = np.linspace(min_t, max_t, 1000)
        pert = perturbation_signal(pert_t, amplitude, phase, mean_val, freq)
        # Plot data and the ideal perturbation signal together
        if not transpose:
            axes.plot(wba_t, wba, color=genotype_color(group_name), label='R-L (rad)')
            axes.plot(pert_t, pert, color='g', label='Perturbation')
            axes.set_xlim((min_t - (max_t - min_t) / 20,
                           max_t + (max_t - min_t) / 20))
            axes.set_ylim((min_wba - (max_wba - min_wba) / 20,
                           max_wba + (max_wba - min_wba) / 20))
        else:
            axes.plot(wba, wba_t, color=genotype_color(group_name), label='R-L (rad)')
            axes.plot(pert, pert_t, color='g', label='Perturbation')
            axes.set_ylim((min_t - (max_t - min_t) / 20,
                           max_t + (max_t - min_t) / 20))
            axes.set_xlim((min_wba - (max_wba - min_wba) / 20,
                           max_wba + (max_wba - min_wba) / 20))

    print('Reading the data...')
    df = perturbation_data_to_records(remove_silences=False)

    dest_dir = ensure_dir(op.join(DEFAULT_PLOTS_DIR, 'bias_vs_meanwba'))

    for freq, freq_data in df.groupby(('freq',)):
        fig, axes = plt.subplots(1, freq_data.group.nunique(), sharey=True, figsize=(20, 12))
        num_obs = min_num_obs(freq_data)
        min_wbas = []
        max_wbas = []
        for groupnum, (group_name, group_data) in enumerate(freq_data.groupby(('group',))):
            wba_t, wba = nanmean_xy(group_data, num_obs=num_obs)
            min_wbas.append(np.min(wba))
            max_wbas.append(np.max(wba))
            perturbation_amplitude = (max_wbas[0] - min_wbas[0]) / 2
            one_group_plot(axes[groupnum], freq, group_name, wba_t, wba,
                           perturbation_amplitude=perturbation_amplitude)
            axes[groupnum].set_title('silenced' if group_name.endswith('E') else 'control', fontsize=24)
        # Conciliate axis
        min_wba, max_wba = min(min_wbas), max(max_wbas)
        amplitude = max((abs(min_wba), abs(max_wba)))
        amplitude += amplitude / 20.
        for ax in axes:
            if not transpose:
                ax.set_ylim((-amplitude, amplitude))
            else:
                ax.set_xlim((-amplitude, amplitude))
        # Labels and cosmetics
        xlabel = 'time (s)' if not transpose else 'Turn Strength and Direction'
        ylabel = 'time (s)' if transpose else 'Turn Strength and Direction'
        for ax in axes:
            ax.set_xlabel(xlabel, fontsize=20)
            ax.legend()
            if transpose:
                ax.axvline(linewidth=4, color='g')
            else:
                ax.axhline(linewidth=4, color='g')
        axes[0].set_ylabel(ylabel, fontsize=20)
        # plt.setp([ax.get_yticklabels() for ax in axes[1:]], visible=False)
        fig.suptitle('%s average for perturbation $\omega$ = %.1f rad/s' % (column_to_mean, freq), fontsize=24)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        dest_file = op.join(dest_dir, '%s__%.1f.%s' % (column_to_mean, freq, img_format))
        FigureCanvasAgg(fig).print_figure(dest_file, dpi=100)

        # plt.show()
        plt.close(fig)

# TODO: actually do it with the gradient of ga
#       (hardcoded, better way pass a function that extracts the desired info
#        and gives name and units to whaterver needs to be plotted)
# bias_vs_meanwba_plots(column_to_mean='ga', gradient=False)
# exit(69)


def text_hpd_report(varname='phase'):
    """Quick and dirty generation of a report of HPDs for all the variables and models and..."""
    results = all_computed_results()
    for (model_id, freq), data in results.groupby(('model_id', 'freq')):
        print 'Model: %s Frequency: %.1f rad/s' % (model_id, freq)
        print '%s 95%% Highest Posterior Density' % varname
        if len(data) != 2:
            print '\tERROR, NUMBER OF GROUPS IS %d, SHOULD BE 2' % len(data)
            continue
        control = MCMCRunManager(data[data.genotype == 'VT37804_TNTin'].iloc[0]['path'])
        silenced = MCMCRunManager(data[data.genotype == 'VT37804_TNTE'].iloc[0]['path'])
        def varnames(result, varname):
            varname_suffix = '_' + varname
            hyperfly, hyperfly_variables, flies, flies_variables = flies_and_variables(result)
            hvar = hyperfly + varname_suffix if varname in set(hyperfly_variables) else None
            fvars = [fly + varname_suffix for fly in flies] if varname in set(flies_variables) else None
            return hvar, fvars
        chvar, cfvars = varnames(control, varname)
        shvar, sfvars = varnames(silenced, varname)

        chtrace = control.traces(chvar).ravel()   # N.B. we just merge all chains
        shtrace = silenced.traces(shvar).ravel()
        cftraces = {cfvar: control.traces(cfvar).ravel() for cfvar in cfvars}
        sftraces = {sfvar: silenced.traces(sfvar).ravel() for sfvar in sfvars}

        print 'control hyperfly \t', hpd(chtrace, 0.05)
        print 'silenced hyperfly\t', hpd(shtrace, 0.05)
        print '*' * 40
        print 'Control flies...'
        print '\tcontrol hyperfly \t', hpd(chtrace, 0.05)
        for flyid, trace in sorted(cftraces.items()):
            print '\t%s\t' % flyid.split('_')[0], hpd(trace, 0.05)
        print '*' * 40
        print 'Silenced flies...'
        print '\tsilenced hyperfly\t', hpd(shtrace, 0.05)
        for flyid, trace in sorted(sftraces.items()):
            print '\t%s\t' % flyid.split('_')[0], hpd(trace, 0.05)
        print '-' * 80


# Box-plot of individual flies

# Autocorrelation of group estimates

# CDE/KDE or histogram of traces

# Trace

# Real data over fitted curve and vs ideal curve

# BODE Plots

def bodelike_plot(model_id='gpa3',
                  varname='phase', varname_first=None,
                  num_chains=4, takelast=10000,
                  alpha=0.05,
                  plot_control=True, plot_silenced=True, img_format='png',
                  show=False):

    mpl_params()

    if varname_first is None:
        varname_first = model_id not in {'gp1', 'gp2', 'gpa1', 'gpa2'}  # Models with old variable naming scheme

    def varnames(result, varname, varname_first=varname_first):
        hyperfly, hyperfly_variables, flies, flies_variables = \
            flies_and_variables(result, varname_first=varname_first)
        if varname_first:
            varname_prefix = varname + '_'
            hvar = varname_prefix + hyperfly if varname in set(hyperfly_variables) else None
            fvars = [varname_prefix + fly for fly in flies] if varname in set(flies_variables) else None
        else:
            varname_suffix = '_' + varname
            hvar = hyperfly + varname_suffix if varname in set(hyperfly_variables) else None
            fvars = [fly + varname_suffix for fly in flies] if varname in set(flies_variables) else None
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
    # Collect and mix traces for all frequencies
    for (model_id, freq), data in results.groupby(('model_id', 'freq')):
        print freq
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
    # Plot the big thingie
    if plot_control:
        plt.plot(np.hstack([ctraces[freq] for freq in freqs]), color='b', label='control')
    if plot_silenced:
        plt.plot(np.hstack([straces[freq] for freq in freqs]), color='r', label='silenced')
    for i, freq in enumerate(freqs):
        xmin = num_chains * takelast * i
        xmax = num_chains * takelast * (i + 1)
        plt.axvline(x=xmax, color='k')
        plt.plot((xmin, xmax), [chpds[i][0]] * 2, color='c', linewidth=4)
        plt.plot((xmin, xmax), [chpds[i][1]] * 2, color='c', linewidth=4)
        plt.plot((xmin, xmax), [shpds[i][0]] * 2, color='m', linewidth=4)
        plt.plot((xmin, xmax), [shpds[i][1]] * 2, color='m', linewidth=4)
        # Gelman-Rubin R^2
        print '\t%s %s control freq %.1f; GR=%.2f' % (model_id, varname, freq,
                                                      gelman_rubin(ctraces[freq].reshape(num_chains, -1)))
        print '\t%s %s silence freq %.1f; GR=%.2f' % (model_id, varname, freq,
                                                      gelman_rubin(straces[freq].reshape(num_chains, -1)))
        # TODO: Geweke, autocorr, put graphically in the plot
    plt.title('Model: %s; Variable: %s' % (model_id, varname))
    plt.xlabel('$\omega$')
    plt.ylabel('%s' % varname)
    plt.tick_params(axis='x',           # changes apply to the x-axis
                    which='both',       # both major and minor ticks are affected
                    bottom='off',       # ticks along the bottom edge are off
                    top='off',          # ticks along the top edge are off
                    labelbottom='off')  # labels along the bottom edge are off
    plt.legend()
    plt.tight_layout()
    plt.savefig(op.join(home, '%s-%s.%s' % (model_id, varname, img_format)))
    if show:
        plt.show()


def failsafe_plot(model, var, fignum=0, show=False):
    if var == 'amplitude' and not 'gpa' in model:
        return
    print model, var
    try:
        plt.figure(fignum)
        bodelike_plot(model_id=model, varname=var, show=show)
        plt.close(fignum)
    except Exception, e:
        print '\tFailed: %s' % str(e)

# # models = ('gpa1', 'gpa2', 'gpa3', 'gpa3nomap', 'gpa4', 'gp1', 'gp2')
# models = ('gp1', 'gpa3', 'gpa3nomap')
# # models = ('gpa4',)
# # models = ('gpa2',)
# # models = ('gp2',)
# hypervars = ('amplitude', 'phase')
# Parallel(n_jobs=1)(delayed(failsafe_plot)(model, var, show=False, fignum=i)
#                    for i, (model, var) in enumerate(product(models, hypervars)))
#
# exit(71)
#

###########################################
# DC vs Amplitude
###########################################


###########################################
# ENTRY POINT
###########################################


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([bias_vs_wba_plots, bias_vs_meanwba_plots])
    parser.dispatch()

#
###########################################
# ADAPTING PYMC PLOTS
###########################################
# So how much data do we have?
#   - Do we have n flies?
#   - Or do we have n cycles?
#   - Or do we have n observations?
#
###########################################
# ADAPTING PYMC PLOTS
###########################################
#
#
# from pymc.Matplot import var_str
# import six
#
# def summary_plot(
#     pymc_obj, name='model', format='png', suffix='-summary', path='./',
#     alpha=0.05, quartiles=True, hpd=True, rhat=True, main=None, xlab=None, x_range=None,
#         custom_labels=None, chain_spacing=0.05, vline_pos=0):
#     """
#     Model summary plot
#
#     Generates a "forest plot" of 100*(1-alpha)% credible intervals for either the
#     set of nodes in a given model, or a specified set of nodes.
#
#     :Arguments:
#         pymc_obj: PyMC object, trace or array
#             A trace from an MCMC sample or a PyMC object with one or more traces.
#
#         name (optional): string
#             The name of the object.
#
#         format (optional): string
#             Graphic output format (defaults to png).
#
#         suffix (optional): string
#             Filename suffix.
#
#         path (optional): string
#             Specifies location for saving plots (defaults to local directory).
#
#         alpha (optional): float
#             Alpha value for (1-alpha)*100% credible intervals (defaults to 0.05).
#
#         quartiles (optional): bool
#             Flag for plotting the interquartile range, in addition to the
#             (1-alpha)*100% intervals (defaults to True).
#
#         hpd (optional): bool
#             Flag for plotting the highest probability density (HPD) interval
#             instead of the central (1-alpha)*100% interval (defaults to True).
#
#         rhat (optional): bool
#             Flag for plotting Gelman-Rubin statistics. Requires 2 or more
#             chains (defaults to True).
#
#         main (optional): string
#             Title for main plot. Passing False results in titles being
#             suppressed; passing False (default) results in default titles.
#
#         xlab (optional): string
#             Label for x-axis. Defaults to no label
#
#         x_range (optional): list or tuple
#             Range for x-axis. Defaults to matplotlib's best guess.
#
#         custom_labels (optional): list
#             User-defined labels for each node. If not provided, the node
#             __name__ attributes are used.
#
#         chain_spacing (optional): float
#             Plot spacing between chains (defaults to 0.05).
#
#         vline_pos (optional): numeric
#             Location of vertical reference line (defaults to 0).
#
#     """
#     try:
#         import matplotlib.gridspec as gridspec
#     except ImportError:
#         gridspec = None
#     from pylab import  plot as pyplot, xlabel, xlim, ylim, savefig
#     from pylab import subplot, axvline, yticks, xticks
#     from pylab import title
#     from pylab import errorbar
#
#
#     # Quantiles to be calculated
#     quantiles = [100 * alpha / 2, 50, 100 * (1 - alpha / 2)]
#     if quartiles:
#         quantiles = [100 * alpha / 2, 25, 50, 75, 100 * (1 - alpha / 2)]
#
#     # Range for x-axis
#     plotrange = None
#
#     # Number of chains
#     chains = None
#
#     # Gridspec
#     gs = None
#
#     # Subplots
#     interval_plot = None
#     rhat_plot = None
#
#     try:
#         # First try Model type
#         vars = pymc_obj._variables_to_tally
#
#     except AttributeError:
#
#         try:
#
#             # Try a database object
#             vars = pymc_obj._traces
#
#         except AttributeError:
#
#             # Assume an iterable
#             vars = pymc_obj
#
#     from pymc import gelman_rubin
#     from pymc.utils import quantiles as calc_quantiles, hpd as calc_hpd
#
#     # Calculate G-R diagnostics
#     if rhat:
#         try:
#             R = gelman_rubin(pymc_obj)
#         except (ValueError, TypeError):
#             try:
#                 R = {}
#                 for variable in vars:
#                     R[variable.__name__] = gelman_rubin(variable)
#             except ValueError:
#                 print(
#                     'Could not calculate Gelman-Rubin statistics. Requires multiple chains of equal length.')
#                 rhat = False
#
#     # Empty list for y-axis labels
#     labels = []
#     # Counter for current variable
#     var = 1
#
#     for variable in vars:
#
#         # Extract name
#         varname = 'kkvaca'
#
#         # Retrieve trace(s)
#         i = 0
#         traces = []
#         while True:
#             try:
#                 # traces.append(pymc_obj.trace(varname, chain=i)[:])
#                 traces.append(variable.trace(chain=i))
#                 i += 1
#             except (KeyError, IndexError):
#                 break
#
#         chains = len(traces)
#
#         if gs is None:
#             # Initialize plot
#             if rhat and chains > 1:
#                 gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
#
#             else:
#
#                 gs = gridspec.GridSpec(1, 1)
#
#             # Subplot for confidence intervals
#             interval_plot = subplot(gs[0])
#
#         # Get quantiles
#         data = [calc_quantiles(d, quantiles) for d in traces]
#         if hpd:
#             # Substitute HPD interval
#             for i, d in enumerate(traces):
#                 hpd_interval = calc_hpd(d, alpha).T
#                 data[i][quantiles[0]] = hpd_interval[0]
#                 data[i][quantiles[-1]] = hpd_interval[1]
#
#         data = [[d[q] for q in quantiles] for d in data]
#         # Ensure x-axis contains range of current interval
#         if plotrange:
#             plotrange = [min(
#                          plotrange[0],
#                          nmin(data)),
#                          max(plotrange[1],
#                              nmax(data))]
#         else:
#             plotrange = [nmin(data), nmax(data)]
#
#         try:
#             # First try missing-value stochastic
#             value = variable.get_stoch_value()
#         except AttributeError:
#             # All other variable types
#             value = variable.value
#
#         # Number of elements in current variable
#         k = size(value)
#
#         # Append variable name(s) to list
#         if k > 1:
#             names = var_str(varname, shape(value))
#             labels += names
#         else:
#             labels.append(varname)
#             # labels.append('\n'.join(varname.split('_')))
#
#         # Add spacing for each chain, if more than one
#         e = [0] + [(chain_spacing * ((i + 2) / 2)) * (
#             -1) ** i for i in range(chains - 1)]
#
#         # Loop over chains
#         for j, quants in enumerate(data):
#
#             # Deal with multivariate nodes
#             if k > 1:
#
#                 for i, q in enumerate(transpose(quants)):
#
#                     # Y coordinate with jitter
#                     y = -(var + i) + e[j]
#
#                     if quartiles:
#                         # Plot median
#                         pyplot(q[2], y, 'bo', markersize=4)
#                         # Plot quartile interval
#                         errorbar(
#                             x=(q[1],
#                                 q[3]),
#                             y=(y,
#                                 y),
#                             linewidth=2,
#                             color="blue")
#
#                     else:
#                         # Plot median
#                         pyplot(q[1], y, 'bo', markersize=4)
#
#                     # Plot outer interval
#                     errorbar(
#                         x=(q[0],
#                             q[-1]),
#                         y=(y,
#                             y),
#                         linewidth=1,
#                         color="blue")
#
#             else:
#
#                 # Y coordinate with jitter
#                 y = -var + e[j]
#
#                 if quartiles:
#                     # Plot median
#                     pyplot(quants[2], y, 'bo', markersize=4)
#                     # Plot quartile interval
#                     errorbar(
#                         x=(quants[1],
#                             quants[3]),
#                         y=(y,
#                             y),
#                         linewidth=2,
#                         color="blue")
#                 else:
#                     # Plot median
#                     pyplot(quants[1], y, 'bo', markersize=4)
#
#                 # Plot outer interval
#                 errorbar(
#                     x=(quants[0],
#                         quants[-1]),
#                     y=(y,
#                         y),
#                     linewidth=1,
#                     color="blue")
#
#         # Increment index
#         var += k
#
#     if custom_labels is not None:
#         labels = custom_labels
#
#     # Update margins
#     left_margin = max([len(x) for x in labels]) * 0.015
#     gs.update(left=left_margin, right=0.95, top=0.9, bottom=0.05)
#
#     # Define range of y-axis
#     ylim(-var + 0.5, -0.5)
#
#     datarange = plotrange[1] - plotrange[0]
#     xlim(plotrange[0] - 0.05 * datarange, plotrange[1] + 0.05 * datarange)
#
#     # Add variable labels
#     yticks([-(l + 1) for l in range(len(labels))], labels)
#
#     # Add title
#     if main is not False:
#         plot_title = main or str(int((
#             1 - alpha) * 100)) + "% Credible Intervals"
#         title(plot_title)
#
#     # Add x-axis label
#     if xlab is not None:
#         xlabel(xlab)
#
#     # Constrain to specified range
#     if x_range is not None:
#         xlim(*x_range)
#
#     # Remove ticklines on y-axes
#     for ticks in interval_plot.yaxis.get_major_ticks():
#         ticks.tick1On = False
#         ticks.tick2On = False
#
#     for loc, spine in six.iteritems(interval_plot.spines):
#         if loc in ['bottom', 'top']:
#             pass
#             # spine.set_position(('outward',10)) # outward by 10 points
#         elif loc in ['left', 'right']:
#             spine.set_color('none')  # don't draw spine
#
#     # Reference line
#     axvline(vline_pos, color='k', linestyle='--')
#
#     # Genenerate Gelman-Rubin plot
#     if rhat and chains > 1:
#
#         # If there are multiple chains, calculate R-hat
#         rhat_plot = subplot(gs[1])
#
#         if main is not False:
#             title("R-hat")
#
#         # Set x range
#         xlim(0.9, 2.1)
#
#         # X axis labels
#         xticks((1.0, 1.5, 2.0), ("1", "1.5", "2+"))
#         yticks([-(l + 1) for l in range(len(labels))], "")
#
#         i = 1
#         for variable in vars:
#
#             if variable._plot == False:
#                 continue
#
#             # Extract name
#             varname = variable.__name__
#
#             try:
#                 value = variable.get_stoch_value()
#             except AttributeError:
#                 value = variable.value
#
#             k = size(value)
#
#             if k > 1:
#                 pyplot([min(r, 2) for r in R[varname]], [-(j + i)
#                        for j in range(k)], 'bo', markersize=4)
#             else:
#                 pyplot(min(R[varname], 2), -i, 'bo', markersize=4)
#
#             i += k
#
#         # Define range of y-axis
#         ylim(-i + 0.5, -0.5)
#
#         # Remove ticklines on y-axes
#         for ticks in rhat_plot.yaxis.get_major_ticks():
#             ticks.tick1On = False
#             ticks.tick2On = False
#
#         for loc, spine in six.iteritems(rhat_plot.spines):
#             if loc in ['bottom', 'top']:
#                 pass
#                 # spine.set_position(('outward',10)) # outward by 10 points
#             elif loc in ['left', 'right']:
#                 spine.set_color('none')  # don't draw spine
#
#     savefig("%s%s%s.%s" % (path, name, suffix, format))
#
#
# def plot_summary_report(varname='phase'):
#     results = all_computed_results()
#     pairs = results.groupby(('model_id', 'freq'))
#     for (model_id, freq), data in pairs:
#         print 'Model: %s Frequency: %.1f rad/s' % (model_id, freq)
#         print '%s 95%% Highest Posterior Density' % varname
#         if len(data) != 2:
#             print '\tERROR, NUMBER OF GROUPS IS %d, SHOULD BE 2' % len(data)
#             continue
#         control = MCMCRunManager(data[data.genotype == 'VT37804_TNTin'].iloc[0]['path'])
#         silenced = MCMCRunManager(data[data.genotype == 'VT37804_TNTE'].iloc[0]['path'])
#         def varnames(result, varname):
#             varname_suffix = '_' + varname
#             hyperfly, hyperfly_variables, flies, flies_variables = flies_and_variables(result)
#             hvar = hyperfly + varname_suffix if varname in set(hyperfly_variables) else None
#             fvars = [fly + varname_suffix for fly in flies] if varname in set(flies_variables) else None
#             return hvar, fvars
#         chvar, cfvars = varnames(control, varname)
#         shvar, sfvars = varnames(silenced, varname)
#
#         chtrace = control.pymctraces(chvar)
#         shtrace = silenced.pymctraces(shvar)
#         # cftraces = {cfvar: control.traces(cfvar).ravel() for cfvar in cfvars}
#         # sftraces = {sfvar: silenced.traces(sfvar).ravel() for sfvar in sfvars}
#
#         summary_plot((chtrace, shtrace), name='mola', path=op.expanduser('~'))
#