import numpy as np


def _argrelmax(x):
    '''
    Find the relative maxima of 1D time series data.

    Parameters
    ----------
    x : array-like
        Time series data.

    Returns
    -------
    peaks : array-like
        Return the True values for all peaks.
    '''
    peaks = np.r_[False, x[1:] > x[:-1]] & np.r_[x[:-1] >= x[1:], False]
    return peaks


def _argupcross(x, x_up):
    '''
    Find the cross ups of 1D time series data.

    Parameters
    ----------
    x : array-like
        Time series data.
    x_up : float
        Upcrossing value.

    Returns
    -------
    zeroups : array-like
        Return the True values for all upcrossings.
    '''
    zeroups = np.r_[x[1:] > x_up, False] & np.r_[x <= x_up]
    return zeroups


def argrelmax(x):
    '''
    Find the relative maxima of 1D time series data.

    Parameters
    ----------
    x : array-like
        Time series data.

    Returns
    -------
    peaks : array-like
        Return the index of all peaks. If no peak is found, the index of the
        largest value is returned.

    Notes
    -----
    Similar to scipy.signal.argrelmax but significantly faster.
    '''
    peaks_bool = _argrelmax(x)
    peaks = np.flatnonzero(peaks_bool)
    if peaks.size:
        return peaks
    else:
        return np.asarray([np.argmax(x)])


def argupcross(x, x_up=0.):
    '''
    Find the upcrossing of 1D time series data.

    Parameters
    ----------
    x : array-like
        Time series data.
    x_up : float, optional
        Upcrossing value. Default is 0.

    Returns
    -------
    zeroups : array-like
        Return the index of all values just before an upcrossing. If no
        upcrossings are found, the index of the first value is returned.
    '''
    zeroups_bool = _argupcross(x, x_up=x_up)
    zeroups = np.flatnonzero(zeroups_bool)
    if zeroups.size:
        return zeroups
    else:
        return np.array([0])


def argrelmax_decluster(x, x_up=0.):
    '''
    Find the declustred relative maxima of 1D time series data.

    Parameters
    ----------
    x : array-like
        Time series data.
    x_up : float, optional
        Upcrossing value. Default is 0.

    Returns
    -------
    peaks : array-like
        Return the index of largest peaks between two upcrossing. If no peak or
        upcrossing pair is found, the index of the largest value is returned.
    '''
    zeroups = argupcross(x, x_up)
    x_sub = np.split(x, zeroups)
    peaks = np.asarray(
        [x_len + np.argmax(x) for x_len, x in zip(zeroups, x_sub[1:-1])])

    if peaks.size:
        return peaks
    else:
        return np.asarray([np.max([np.argmax(x), x_up])])
