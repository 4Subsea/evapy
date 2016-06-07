import numpy as np


def _argrelmax(x):
    '''
    Find the relative maxima of 1D time series data.

    Parameters
    ----------
    od_pipe : array-like
        Time series data.

    Returns
    -------
    peaks : array-like
        Return the index of all peaks.
    '''
    peaks = np.r_[False, x[1:] > x[:-1]] & np.r_[x[:-1] > x[1:], False]
    return peaks


def _argzeroup(x, x_up=0.):
    '''
    Calculate the zero up-crossing of 1D data.
    '''
    zeroups = np.r_[x[1:] > 0., False] & np.r_[x < 0.]
    return zeroups


def argrelmax(x):
    '''
    Calculate the relative maxima of 1D data.
    '''
    peaks = _argrelmax(x)
    if peaks.size:
        return np.flatnonzero(peaks)
    else:
        return np.asarray(np.argmax(x))


def argzeroup(x):
    '''
    Calculate the zero up-crossing of 1D data.
    '''
    zeroups = _argzeroup(x)
    if zeroups.size:
        return np.flatnonzero(zeroups)
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