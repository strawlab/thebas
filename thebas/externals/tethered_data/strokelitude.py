# coding=utf-8
"""
Convenient access and best practices for dealing with strokelitude data.

See https://github.com/motmot/strokelitude
"""
from __future__ import division, print_function
import pandas
import h5py
import numpy as np


class Strokelitude(object):
    """Convenience class to access strokelitude data, collected at tethered stations."""
    def __init__(self, h5file):
        super(Strokelitude, self).__init__()
        self.h5 = h5file

    def _attr(self, attr):
        with h5py.File(self.h5) as reader:
            return reader['strokelitude'][attr]

    #### Strokelitude info

    def rwa(self):
        """Returns the right wing angle, in radias, as a 1D numpy array."""
        return self._attr('right_wing_angle_radians')

    def lwa(self):
        """Returns the left wing angle, in radias, as a 1D numpy array."""
        return self._attr('left_wing_angle_radians')

    def r_l(self):
        """Returns the right wing angle - left wing angle, in radias, as a 1D numpy array."""
        return self.rwa() - self.lwa()

    def wba(self):
        """Returns the wingbeat amplitude (radians), defined as right wing angle - left wing angle."""
        return self.r_l()

    def lli(self):
        """Returns the left legbox intensity."""
        return self._attr('left_legbox_intensity')

    def rli(self):
        """Returns the right legbox intensity."""
        return self._attr('right_legbox_intensity')

    def lwi(self):
        """Returns the left wingbox intensity."""
        return self._attr('left_wingbox_intensity')

    def rwi(self):
        """Returns the right wingbox intensity."""
        return self._attr('right_wingbox_intensity')

    def t(self):
        """Returns the timestamps in seconds."""
        return self._attr('t_secs') + self._attr('t_nsecs') * 1E-9

    def ts(self):
        """Returns a tuple (t_secs, t_nsecs)."""
        return self._attr('t_secs'), self._attr('t_nsecs')

    #### Info from ros header msgs

    def hseq(self):
        """Returns the header seq."""
        return self._attr('header_seq')

    def hstamp(self):
        """Returns the header stamp."""
        return self._attr('header_stamp')

    def hstampsecs(self):
        """Returns the header stamp (seconds part)."""
        return self._attr('header_stamp_secs')

    def hstampnsecs(self):
        """Returns the header stamp (nanoseconds part)."""
        return self._attr('header_stamp_nsecs')

    def frameid(self):
        """Returns the header frame id."""
        return self._attr('header_frame_id')

    #### Convenience functions

    def topandas(self):
        """Returns a pandas dataframe with a selection of strokelitude data for further analysis."""
        with h5py.File(self.h5) as hdf:
            strokelitude = {
                'LWA': hdf['strokelitude']['left_wing_angle_radians'],
                'RWA': hdf['strokelitude']['right_wing_angle_radians'],
                'LeftLegbox': hdf['strokelitude']['left_legbox_intensity'],
                'RightLegbox': hdf['strokelitude']['right_legbox_intensity'],
                'LeftWingbox': hdf['strokelitude']['left_wingbox_intensity'],
                'RightWingbox': hdf['strokelitude']['right_wingbox_intensity'],
                'RelativeTime': hdf['strokelitude']['t']
            }
            strokelitude['WBA'] = strokelitude['LWA'] - strokelitude['RWA']
            return pandas.DataFrame(strokelitude)

######
# Preprocessing stuff.
# These functions are applied over strokelitude data and possible stimuli data
# (e.g. silence periods must be removed on stimuli too).
######


def detect_noflight(ri, li, threshold_factor=1.035, safety_range_size=100):
    """Returns a numpy mask where the true elements indicate a no-flight (aka silence) period.

    Parameters:
      - ri, li: numpy arrays of left and right wingbox intensities
      - threshold_factor: multiplies the medians of ri and li to find the threshold on intensities
      - safety_range_size: if greater than 0, include (-size, +size) neighbor measurements in the silence region.
                           Accounts for "landings" and "taking-offs".
                           1 second (100 measurements at 100Hz) should still be conservative
                             for the sine-perturbation experiment.
                           2 seconds (200 measurements at 100Hz) should be fine
                             for the stripe-based experiments
                           (Lisa believes they might need more time to fix the stripe than to respond to optomotor...)
    """
    threshold_right = threshold_factor * np.median(ri)
    threshold_left = threshold_factor * np.median(li)

    silences_mask = (ri > threshold_right) & (li > threshold_left)

    if safety_range_size > 0:
        for silence in np.where(silences_mask)[0]:
            silences_mask[max(0, silence - safety_range_size):silence + safety_range_size] = True

    return silences_mask


def stigmatize_noflight(noflight_mask, stigma=np.nan, inplace=False, *signals):
    """Returns an iterator assigning all the elements specified in the mask to the "stigma", for each of the signals."""
    for signal in signals:
        if not inplace:
            signal = signal.copy()
        signal[noflight_mask] = stigma
        yield signal


def remove_noflight(noflight_mask, *signals):
    """Returns an iterator over the signals without the elements marked on "noflight_mask"."""
    for signal in signals:
        yield signal[~noflight_mask]


def filter_signals_gaussian(sigma=10):
    """Returns a function that smoothes signals using a Gaussian kernel with the specified sigma."""
    from scipy.ndimage.filters import gaussian_filter1d as gaussian
    return lambda *signals: (gaussian(signal, sigma=sigma) for signal in signals)


def filter_signals_lowpass(dt_seconds=0.01, cutoff_hz=20.0, butter_order=8):
    """Returns a function that smoothes signals using a lowpass filter with the specified parameters."""
    from scipy.signal import butter, filtfilt
    sample_rate_hz = 1.0 / dt_seconds
    nyquist_rate_hz = sample_rate_hz * 0.5
    filt_b, filt_a = butter(butter_order, cutoff_hz / nyquist_rate_hz)
    return lambda *signals: (filtfilt(filt_b, filt_a, signal) for signal in signals)