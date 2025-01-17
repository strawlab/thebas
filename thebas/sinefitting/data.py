# coding=utf-8
"""Data munging from the sisusoidal-perturbation experiments, focused on data-preparation for Bayesian modelling.

THE method you probably want to use is this (at the end of the file):
  perturbation_data_to_records
Look at the documentation to understand how to access the data.
"main" (at the end of the file) shows some examples
"""
from glob import glob
import os
import os.path as op

import h5py
import numpy as np
import pandas as pd
from pandas import DataFrame

from thebas.externals.tethered_data.misc import rostimearr2datetimeidx
from thebas.externals.tethered_data.strokelitude import Strokelitude, filter_signals_gaussian, filter_signals_lowpass, \
    detect_noflight, remove_noflight
from thebas.sinefitting import DCN_TEST_HDF5, HS_PROJECT


####################################
# Convenient access to the data as provided by strokelitude
####################################
from thebas.sinefitting import DCN_PROJECT


class TetheredSinewaveData(object):
    def __init__(self, h5file):
        super(TetheredSinewaveData, self).__init__()
        self.h5 = h5file

    def bias(self):
        """Returns the bias info (essentially the sinewave over time) as 1D numpy arrays.
        It gives 4 arrays: amplitude, the time in seconds and nanoseconds and the sum of the two times.

        These are virtual-reality-controler measurements (higher frequency)

        Example:
        >>> bias, btsec, bnsec, bt = TetheredSinewaveData(DCN_TEST_HDF5).bias()
        """
        with h5py.File(self.h5) as reader:
            bias = reader['bias']
            return bias['data'], \
                bias['t_secs'], bias['t_nsecs'], bias['t']

    def bias2pandas(self):
        """Returns a pandas DataFrame with the bias in a column indexed by relative time and starting from 0,
        together with the actual start of the bias to be put on place."""
        bias, _, _, bt = self.bias()
        exp_start = bt[0]
        t_bias = rostimearr2datetimeidx(floattime=bt, offset=exp_start)
        return DataFrame(bias['data'], index=t_bias, columns=['bias']), exp_start

    def nfga(self):
        """Returns the "new figure ground angles" information, in radians.

        These are sensed quantities (lower frequency).

        Example:
        >>> fa, ga, fon, gon, ts, tns, t = TetheredSinewaveData(DCN_TEST_HDF5).nfga()
        """
        with h5py.File(self.h5) as reader:
            nfga = reader['new_figure_ground_angles']
            return nfga['figure_angle_radians'], \
                nfga['ground_angle_radians'], \
                nfga['figure_on'], \
                nfga['ground_on'], \
                nfga['t_secs'], nfga['t_nsecs'], nfga['t']

    def nfgm(self):
        """Returns the "new figure ground models" information.

        These are the virtual-reality configuration files.

        Example:
        >>> fmf, gmf, ts, tns, t = TetheredSinewaveData(DCN_TEST_HDF5).nfgm()
        """
        with h5py.File(self.h5) as reader:
            nfgm = reader['new_figure_ground_models']
            return nfgm['figure_model_filename'], nfgm['ground_model_filename'],\
                nfgm['t_secs'], nfgm['t_nsecs'], nfgm['t']


def all_biases(pbproject):
    """A convenience function that returns a dictionary {flyid->(bias, bias_t)} for all the flies in the experiment."""
    flies_dirs = pbproject.all_genotypes_dirs()
    biases = {}
    for root in flies_dirs:
        for hdf5 in glob(op.join(root, '*.hdf5')):
            flyid = op.basename(hdf5)[:-len('.hdf5')]
            bias, _, _, bias_t = TetheredSinewaveData(hdf5).bias()
            biases[flyid] = (bias, bias_t)
    return biases


def perturbation_bias_info():
    """Returns a triplet (freqs, freqs_on_durations, freq_start_stop_times).
    Computes information for the sinusoidal perturbation bias used in these experiments.
    The code is adapted from "lisa_bg_cl_perturbation.py".
    """
    # Perturbation angular frequencies (rad/s) (Recall dtheta/dt = w = 2*pi*f)
    freqs = np.array([0.5, 1, 2, 4, 8, 16, 32, 40])  # FIXME: hardcoded
    # For how long each *angular* frequency is "on" (seconds).
    # Let a frequency run for 8 complete cycles.
    # The alternative could be a fixed ontime length for all freqs.
    # But that could be inconvenient for Lisa and maybe the fly.
    freqs_on_durations = 16 * np.pi / freqs  # FIXME: hardcoded

    freqs_start_stop_times = np.append(0, np.cumsum(freqs_on_durations))
    return freqs, freqs_on_durations, freqs_start_stop_times
    # we might infer this in a general way from the bias signal itself


def read_perturbation_experiment_data(hdf5file, smooth=True, use_lowpass=False, remove_silences=True):
    """Returns a map {freq->(wba, wba_t, ga)}, after smoothing and removing silences."""
    # Load recorded data
    stk = Strokelitude(hdf5file)
    wba = stk.wba()                       # The wingbeat amplitude (well, maybe just R-L)
    lwi, rwi = stk.lwi(), stk.rwi()       # Wingbox intensities, to remove silences
    wba_t = stk.t()                       # Timestamps for wba measurements (seconds)
    dt = np.mean(wba_t[1:] - wba_t[:-1])  # Sampling period (seconds)
    tsd = TetheredSinewaveData(hdf5file)
    _, ga, _, _, _, _, _ = tsd.nfga()     # Ground angles

    # Smooth the data
    if smooth:
        signal_filter = filter_signals_gaussian() if not use_lowpass else filter_signals_lowpass(dt_seconds=dt)
        lwi, rwi, wba = signal_filter(lwi, rwi, wba)

    # Detect not flying periods and *remove* them
    if remove_silences:
        noflight_mask = detect_noflight(lwi, rwi)
        wba_t, wba, ga = remove_noflight(noflight_mask, wba_t, wba, ga)

    # Perturbation bias and relative start/stop times for each frequency
    _, _, _, bias_t = tsd.bias()  # Actually we do not even need the bias at this stage
    freqs, _, freqs_start_stop_times = perturbation_bias_info()

    # The experiment starts and ends when the perturbation-bias is "on"
    exp_start, exp_end = bias_t[0], bias_t[-1]
    samples_with_bias = (exp_start < wba_t) & (wba_t <= exp_end)
    wba = wba[samples_with_bias]
    wba_t = wba_t[samples_with_bias]
    ga = ga[samples_with_bias]

    # Now rebase time onto 0
    # This is correct for this experiment but should be done before silence removal on the general case
    wba_t -= exp_start
    assert wba_t[0] >= 0
    assert wba_t[-1] <= freqs_start_stop_times[-1]

    # Group on the different perturbation frequencies
    frequencies_data = {}
    for i, freq in enumerate(freqs):
        freqon_observations = (freqs_start_stop_times[i] <= wba_t) & \
                              (wba_t <= freqs_start_stop_times[i + 1])
        frequencies_data[freq] = (wba[freqon_observations],
                                  wba_t[freqon_observations],
                                  ga[freqon_observations])
    return frequencies_data


def read_flies(root, remove_silences=True):
    """Returns group_name, {flyname -> (wba, wba_t)}."""
    flies = {op.splitext(op.basename(hdf5))[0]: read_perturbation_experiment_data(hdf5, remove_silences=remove_silences)
             for hdf5 in glob(op.join(root, '*.hdf5'))}
    return op.basename(root), flies


def all_perturbation_data(pbproject, remove_silences=True):
    """{group_name -> {flyname -> {freq -> (wba, wba_t, ga)}}}"""
    return dict([read_flies(root, remove_silences=remove_silences) for root in pbproject.all_genotypes_dirs()])


def print_data_sizes_summary(pbproject=HS_PROJECT, data=None):
    """Prints a summary of data sizes.
    Just an example of how to use all_perturbation_data.
    """
    if data is None:
        data = all_perturbation_data(pbproject)
    for group, flies in data.iteritems():
        print('%s flies data sizes...' % group)
        for fly, frequency_groups in flies.iteritems():
            print '\tFly:', fly
            for frequency, (wba, wba_t, ga) in sorted(frequency_groups.items()):
                print('\t\tFrequency=%.1f\t NumObservations=%d' % (frequency, len(wba)))
        print('*' * 80)


####################################
# A record per (fly, genotype, frequency) makes for the most sensible access scheme
####################################

# ---- library independent HDF5 format (to forget about pickles / pytables and anything less standard/durable)

def save_perturbation_record_data_to_hdf5(data, hdf_file, overwrite=False):
    if op.isfile(hdf_file):
        if overwrite:
            os.unlink(hdf_file)
        else:
            raise Exception('%s already exists and overwrite is False' % hdf_file)
    with h5py.File(hdf_file, 'w-') as h5:
        for _, row in data.iterrows():
            exp = h5.require_group(row['flyid'])
            exp.attrs['genotype'] = row['genotype']
            freq = exp.require_group('freq=%g' % row['freq'])
            freq.attrs['freq'] = row['freq']
            freq.create_dataset('wba_t', data=row['wba_t'], compression='lzf')
            freq.create_dataset('wba', data=row['wba'], compression='lzf')
            freq.create_dataset('ga', data=row['ga'], compression='lzf')
            freq.attrs['ideal_start'] = row['ideal_start']
            freq.attrs['ideal_stop'] = row['ideal_stop']
            freq.attrs['ideal_numobs'] = row['ideal_numobs']
            freq.attrs['numobs'] = row['numobs']


def load_perturbation_record_data_from_hdf5(hdf_file):
    all_records = []
    with h5py.File(hdf_file, 'r') as h5:
        for flyid, flydata in h5.iteritems():
            genotype = flydata.attrs['genotype']
            for _, freqdata in flydata.iteritems():
                all_records.append((
                    flyid,
                    genotype,
                    freqdata.attrs['freq'],
                    freqdata['wba_t'][:],
                    freqdata['wba'][:],
                    freqdata['ga'][:],
                    freqdata.attrs['ideal_start'],
                    freqdata.attrs['ideal_stop'],
                    freqdata.attrs['ideal_numobs'],
                    freqdata.attrs['numobs']
                ))
    return pd.DataFrame(data=all_records, columns=('flyid', 'genotype', 'freq',
                                                   'wba_t', 'wba', 'ga',
                                                   'ideal_start', 'ideal_stop', 'ideal_numobs', 'numobs'))


def perturbation_data_to_records(pbproject=HS_PROJECT,
                                 dt=0.01,
                                 cache_to_pickle=False,
                                 overwrite_cached=False,
                                 remove_silences=True):
    """Reads the perturbation experiment data into a pandas dataset, in records with the following columns:
       - flyid: the id of the fly
       - genotype: the genotype of the fly
       - freq: the frequency of the perturbation (rad/s)
       - wba_t: the times of each instantaneous measurement (series of seconds)
       - wba: the R-L wingbeat amplitude (series of rad)
       - ga: the ground-angle (series of rad)
       - ideal_start: when the recording started (in seconds, based on 0-start)
       - ideal_stop: when the recording stopped (in seconds, based on 0-start)
       - ideal_numobs: an approximation to the maximum number of observations
       - numobs: the real number of observations (without not-flying periods)

    N.B. Assumes that the data fits comfortably in memory

    Parameters
    ----------
    pbproject: PBProject object
      indicates where to find the data files
    dt: float, default 0.01
      the sampling period (note that the "ideal" sampling rate was 100Hz)
    cache_to_pickle: boolean, default True
      if True, data will be stored as a pickled dataframe, else it will be stored in a simple hdf5 file
    overwrite_cached: boolean, default False
      if True, a possibly existent cache is overwritten
    remove_silences: boolean, default True
      if True, the silences (periods in which the fly is not flying) is removed

    Returns
    -------
    That pandas dataframe

    Examples
    --------
    >>> data = perturbation_data_to_records(DCN_PROJECT)
    >>> print 'There are %d records (%d flyids x 8 frequencies)' % (len(data), data['flyid'].nunique())
    There are 208 records (26 flyids x 8 frequencies)
    >>> print 'There are %d control records' % np.sum(data['genotype'] == 'VT37804_TNTE')
    There are 112 control records
    >>> print 'There are %d DCN-blocked records' % np.sum(data['genotype'] == 'VT37804_TNTin')
    There are 96 DCN-blocked records
    >>> data2 = perturbation_data_to_records(cache_to_pickle=True)
    >>> len(data) == len(data2)
    True
    >>> set(data.columns) == set(data2.columns)
    True
    >>> row1 = data[(data['flyid'] == '2012-12-18-13-46-42') & (data['freq'] == 8.)]
    >>> row2 = data2[(data2['flyid'] == '2012-12-18-13-46-42') & (data2['freq'] == 8.)]
    >>> len(row1)
    1
    >>> len(row2)
    1
    >>> row1 = row1.iloc[0]  # Extract the series
    >>> row2 = row2.iloc[0]
    >>> np.allclose(row1['wba_t'], row2['wba_t'])
    True
    >>> np.allclose(row1['wba'], row2['wba'])
    True
    >>> np.allclose(row1['ga'], row2['ga'])
    True
    """
    if cache_to_pickle:
        CACHE_FILE = op.join(pbproject.data_root, 'perturbation_bias_munged_data.pickle') \
            if remove_silences else \
            op.join(pbproject.data_root, 'perturbation_bias_munged_data_with_silences.pickle')
    else:
        CACHE_FILE = op.join(pbproject.data_root, 'perturbation_bias_munged_data.h5') \
            if remove_silences else \
            op.join(pbproject.data_root, 'perturbation_bias_munged_data_with_silences.h5')

    # Return cached data
    if op.isfile(CACHE_FILE) and not overwrite_cached:
        if cache_to_pickle:
            return pd.read_pickle(CACHE_FILE)
        return load_perturbation_record_data_from_hdf5(CACHE_FILE)

    # Remunge
    data = all_perturbation_data(pbproject, remove_silences=remove_silences)
    records = []
    for group, flies in data.iteritems():
        for flyname, f2obs in flies.iteritems():
            for freq, (wba, wba_t, ga) in f2obs.iteritems():
                records.append((group, flyname, freq, wba_t, wba, ga))
    df = DataFrame(records, columns=('genotype', 'flyid', 'freq', 'wba_t', 'wba', 'ga'))
    # "Observation period"
    freqs, _, freqs_start_stop_times = perturbation_bias_info()
    f2times = {f: (freqs_start_stop_times[i], freqs_start_stop_times[i + 1]) for i, f in enumerate(freqs)}
    df['ideal_start'] = df.apply(lambda row: f2times[row['freq']][0], axis=1)
    df['ideal_stop'] = df.apply(lambda row: f2times[row['freq']][1], axis=1)
    df['ideal_numobs'] = (df.ideal_stop - df.ideal_start) / dt
    df['numobs'] = df.apply(lambda row: len(row['wba']), axis=1)

    # Explicitly state columns order - workaround of a strange pandas pickling bug
    df = df[[u'flyid',
             u'genotype',
             u'freq',
             u'wba_t',
             u'wba',
             u'ga',
             u'ideal_start',
             u'ideal_stop',
             u'ideal_numobs',
             u'numobs']]

    # Cache
    if cache_to_pickle:
        df.to_pickle(CACHE_FILE)
    else:
        save_perturbation_record_data_to_hdf5(df, CACHE_FILE)

    return df


if __name__ == '__main__':

    GENOTYPES = ('VT37804_TNTE', 'VT37804_TNTin')   # DCN-silenced, control
    FREQS = (.5, 1., 2., 4., 8., 16., 32., 40.)     # angular frequencies (rad/s)

    # Record format + pandas make exploration easier...
    print('Reading the data...')
    df = perturbation_data_to_records()

    # Report average number of observations for each frequency and genotype
    numobs_by_group_and_freq = df.groupby(by=('genotype', 'freq'))['numobs']
    print(numobs_by_group_and_freq.mean())

    for (group, freq), data in df.groupby(('genotype', 'freq')):
        for _, row in data.iterrows():
            print('fly %s (genotype %s) has %d observations for frequency %g' %
                  (row['flyid'], row['genotype'], len(row['wba']), row['freq']))


#
# So how much data do we have?
#   - Do we have n flies?
#   - Or do we have n cycles?
#   - Or do we have n observations?
#