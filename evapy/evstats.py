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
    peaks = np.r_[False, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:], False]
    return peaks


def _argupcross(x, x_up=0.):
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


def argrelmax_decluster(x):
    '''
    Calculate the declustered relative maxima of 1D data.
    '''
    zeroups = _argzeroup(x)
    peaks = _argrelmax(x)
    peaks_series = zeroups | peaks
    
    index = np.arange(len(x))[peaks_series]
    
    index_rel = argrelmax(x[peaks_series])
    return index[index_rel]