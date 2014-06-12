# coding=utf-8
import numpy as np
import pandas


def rostimearr2datetimeidx(floattime=None, nsectime=None, offset=0):
    """Timestamps to datetime index so that pandas understands the time index, possibly translated on time.

    From Lisa/Andreas code.
    """
    if floattime is None and nsectime is not None:
        time = np.array(nsectime, dtype=np.int64) - int(offset)
    elif floattime is not None and nsectime is None:
        time = np.array(floattime*1e9, dtype=np.int64) - int(float(offset)*1e9)
    else:
        raise ValueError('please specify either floattime or nsectime')

    return pandas.DatetimeIndex(time)


