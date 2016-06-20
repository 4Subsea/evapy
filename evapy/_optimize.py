'''
Custom optimization routines for distributions where ML estimators perform
poorly or do not exist.
'''

import numpy as np


def _residual_error(self, theta, x, y_fun, **kwargs):
    '''
    Return special purspose lsq objective error function to minimize.
    Method is overwritten to hook into scipy.stats optimization framework.

    Parameters
    ----------
    self : object
        Distribution object based on rv_continious
    thete : array-like
        List of parameters according to target distribution.
    y_fun: callable
        Function that takes in a y value and tranforms it.

    Keywords
    --------
    Not documented

    Returns
    -------
    error : float
        Return sum of penilized residual error.

        '''
    try:
        loc = theta[-2]
        scale = theta[-1]
        args = tuple(theta[:-2])
    except IndexError:
        raise ValueError("Not enough input arguments.")

    if not self._argcheck(*args) or scale <= 0:
        return inf

    x = np.asarray((x-loc) / scale)
    x.sort()

    if np.isneginf(self.a).all() and np.isinf(self.b).all():
        Nbad = 0
    else:
        cond0 = (x <= self.a) | (self.b <= x)
        Nbad = sum(cond0)
        if Nbad > 0:
            x = x[~cond0]

    N = len(x)

#  Refactor ecdf out into own code at some point.
    f_ecdf = np.array([(i + 1 - 0.3)/(N + 0.4) for i in range(N)])
    error = np.abs(y_fun(f_ecdf) - y_fun(self._cdf(x, *args)))**2.
    return np.sum(error) + Nbad * 10000.
