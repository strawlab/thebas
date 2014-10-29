# coding=utf-8
import os.path as op

from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import matplotlib.pyplot as plt

from thebas.misc import ensure_dir
from thebas.sinefitting import HS_PROJECT, DCN_PROJECT
from thebas.sinefitting.models import perturbation_signal
from thebas.sinefitting.data import perturbation_data_to_records
from thebas.sinefitting.reports import mpl_params, genotype_color


def bias_vs_wba_plots(pbproject=HS_PROJECT, img_format='png', transpose=True, legend=False):
    """Plots the bias and R-L together, one per fly and frequency."""

    def detect_discontinuities(y, threshold):
        return np.where(np.abs(np.diff(y)) >= threshold)[0]

    def mark_discontinuities(x, y, threshold=0.01, use_x=False):
        discontinuities = detect_discontinuities(y if not use_x else x, threshold) + 1
        x = np.insert(x, discontinuities, np.nan)
        y = np.insert(y, discontinuities, np.nan)
        return x, y

    mpl_params()

    print('Reading the data...')
    df = perturbation_data_to_records(pbproject=pbproject)

    by_freq = df[['genotype', 'freq', 'flyid', 'wba', 'wba_t']].groupby(('freq',))

    dest_dir = ensure_dir(op.join(pbproject.plots_dir, 'bias_vs_wba'))
    for freq, freq_data in by_freq:
        min_t = min([wba_t.min() for wba_t in freq_data['wba_t'] if len(wba_t) > 0])
        max_t = max([wba_t.max() for wba_t in freq_data['wba_t'] if len(wba_t) > 0])
        min_wba = min([wba.min() for wba in freq_data['wba'] if len(wba) > 0])
        max_wba = max([wba.max() for wba in freq_data['wba'] if len(wba) > 0])
        for flyid, flydata in freq_data.groupby(('flyid',)):
            print 'Flyid: %s; freq=%g' % (flyid, freq)
            if len(flydata) > 1:
                raise Exception('The flyid %s is not unique!' % flyid)
            # Fly wba trajectory
            flydata = flydata.iloc[0]
            genotype = flydata.genotype
            x, y = np.array(flydata['wba_t']), np.array(flydata['wba'])
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
                plt.plot(x, y, color=genotype_color(flydata['genotype']), label='Fly R-L (rad)')
                plt.plot(pert_t, pert, color='g', label='Perturbation')
                plt.axhline(linewidth=4, color='g')
                plt.xlabel('time (s)')
                plt.ylabel('Strength of turn')
                if legend:
                    plt.legend()
                plt.xlim((min_t - (max_t - min_t) / 10, max_t + (max_t - min_t) / 10))
            else:
                plt.plot(y, x, color=genotype_color(flydata['genotype']), label='Fly R-L (rad)')
                plt.plot(pert, pert_t, color='g', label='Perturbation')
                plt.axvline(linewidth=4, color='g')
                plt.ylabel('time (s)')
                plt.xlabel('Strength of turn')
                if legend:
                    plt.legend()
                plt.ylim((min_t - (max_t - min_t) / 10, max_t + (max_t - min_t) / 10))
            plt.title('fly:%s(%s); freq=%.1f rad/s' %
                      (flyid, genotype.replace('_', '+'), freq))
            fn = '%s-%.1f-%s.%s' % (genotype, freq, flyid, img_format)
            plt.savefig(op.join(dest_dir, fn))
            plt.close()


def bias_vs_meanwba_plots(pbproject=HS_PROJECT, img_format='png', transpose=True, column_to_mean='wba', gradient=False):
    """Averages for each frequency and genotype and plots a wba vs bias-signal plot."""

    # Some cosmetics
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
    df = perturbation_data_to_records(pbproject=pbproject, remove_silences=False)

    dest_dir = ensure_dir(op.join(pbproject.plots_dir, 'bias_vs_meanwba'))

    for freq, freq_data in df.groupby(('freq',)):
        fig, axes = plt.subplots(1, freq_data.genotype.nunique(), sharey=True, figsize=(20, 12))
        num_obs = min_num_obs(freq_data)
        min_wbas = []
        max_wbas = []
        for groupnum, (genotype, group_data) in enumerate(freq_data.groupby(('genotype',))):
            wba_t, wba = nanmean_xy(group_data, num_obs=num_obs)
            min_wbas.append(np.min(wba))
            max_wbas.append(np.max(wba))
            perturbation_amplitude = (max_wbas[0] - min_wbas[0]) / 2
            one_group_plot(axes[groupnum], freq, genotype.replace('_', 'x'), wba_t, wba,
                           perturbation_amplitude=perturbation_amplitude)
            axes[groupnum].set_title(genotype.replace('_', 'x'), fontsize=24)
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


if __name__ == '__main__':

    print 'Plotting perturbation vs individual fly responses...'
    bias_vs_wba_plots(DCN_PROJECT)
    bias_vs_wba_plots(HS_PROJECT)

    print 'Plotting perturbation vs averaged fly responses...'
    bias_vs_meanwba_plots(DCN_PROJECT)
    bias_vs_meanwba_plots(HS_PROJECT)

    #
    # FIXME: actually do this with the gradient of ga
    #       (hardcoded, better way pass a function that extracts the desired info
    #        and gives name and units to whatever needs to be plotted)
    # Also Need to tweak labels and others for this...
    #
    # bias_vs_meanwba_plots(DCN_PROJECT, column_to_mean='ga', gradient=False)
    # bias_vs_meanwba_plots(HS_PROJECT, column_to_mean='ga', gradient=False)
    #


##################
# Notes
##################
#
# When Lisa averaged, how did she cope with different timings?
# We do it here do it the naive but maybe incorrect way, just
# assuming we can nanmean all without regard for timing
#
##################
#
# TODO: zoom on higher frequencies to really see the range of change
#       take into account that at higher frequencies amplitude must be much lower
#
##################
