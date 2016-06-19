'''
Notes
-----

The distributions will rely on the `scipy.stats` framework.

Similar to scipy.stats._continous_distns.

Thanks to all SciPy developers for their great work!
'''


from __future__ import division

from scipy import special
from scipy.stats import rv_continuous
from scipy import optimize

from numpy import (exp, log, sqrt, pi, inf)

import numpy as np

#  Special constants
_EULER = 0.577215664901532860606512090082402431042
_ZETA3 = 1.202056903159594285399738161511449990765


class rayleigh_gen(rv_continuous):
    """A Rayleigh continuous random variable.
    
    See scipy.stats.rayleigh for usage and documentation

    Notes
    -----
    The probability density function for `rayleigh` is::

        rayleigh.pdf(r) = r * exp(-r**2/2)

    for ``x >= 0``.

    Copy of scipy.stats.rayleigh. Included for convinience in the evapy
    package.

    """
    def _pdf(self, x):
        return r*exp(-0.5*x**2)

    def _cdf(self, x):
        return -special.expm1(-0.5*x**2)

    def _ppf(self, f):
        return sqrt(-2*special.log1p(-f))

    def _sf(self, x):
        return exp(-0.5*x**2)

    def _isf(self, s):
        return sqrt(-2*log(s))

    def _stats(self):
        val = 4 - pi
        return (sqrt(pi/2), val/2, 2*(pi-3)*sqrt(pi)/val**1.5,
                6*pi/val-16/val**2)

    def _entropy(self):
        return _EULER/2.0 + 1 - 0.5*log(2)
rayleigh = rayleigh_gen(a=0.0, name="rayleigh")


class frechet_r_gen(rv_continuous):
    """A Frechet right (or Weibull minimum) continuous random variable.

    %(before_notes)s

    See Also
    --------
    weibull_min : The same distribution as `frechet_r`.
    frechet_l, weibull_max

    Notes
    -----
    The probability density function for `frechet_r` is::

        frechet_r.pdf(x, c) = c * x**(c-1) * exp(-x**c)

    for ``x > 0``, ``c > 0``.

    `frechet_r` takes ``c`` as a shape parameter.

    %(after_notes)s

    %(example)s

    """
    def _pdf(self, x, c):
        return c*x**(c-1)*exp(-x**c)

    def _logpdf(self, x, c):
        return log(c) + (c-1)*log(x) - x**c

    def _cdf(self, x, c):
        return -special.expm1(-x**c)

    def _ppf(self, f, c):
        return (-special.log1p(-f))**(1.0/c)

    def _munp(self, n, c):
        return special.gamma(1.0+n*1.0/c)

    def _entropy(self, c):
        return -_EULER / c - log(c) + _EULER + 1
weibull = frechet_r_gen(a=0.0, name='weibull')
weibull_min = frechet_r_gen(a=0.0, name='weibull_min')


class gumbel_r_gen(rv_continuous):
    """A right-skewed Gumbel continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel_l, gompertz, genextreme

    Notes
    -----
    The probability density function for `gumbel_r` is::

        gumbel_r.pdf(x) = exp(-(x + exp(-x)))

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.

    %(after_notes)s

    %(example)s

    """
    def _pdf(self, x):
        return exp(self._logpdf(x))

    def _logpdf(self, x):
        return -x - exp(-x)

    def _cdf(self, x):
        return exp(-exp(-x))

    def _logcdf(self, x):
        return -exp(-x)

    def _ppf(self, f):
        return -log(-log(f))

    def _stats(self):
        return _EULER, pi*pi/6.0, 12*sqrt(6)/pi**3 * _ZETA3, 12.0/5

    def _entropy(self):
        return _EULER + 1.
gumbel = gumbel_r_gen(name='gumbel')
gumbel_max = gumbel_r_gen(name='gumbel_max')


class gen_exp_tail_gen(rv_continuous):
    """
    Generlized exponential tail distribution.
    """
    def _pdf(self, x, c, q):
        return c*x**(c-1.)*exp(-x**c + log(q))

    def _logpdf(self, x, c, q):
        return log(c) + (c-1)*log(x) - x**c + log(q)

    def _cdf(self, x, c, q):
        return -special.expm1(-x**c + log(q))

    def _logcdf(self, x, c, q):
        return special.log1p(-np.exp(-x**c + log(q)))

    def _ppf(self, f, c, q):
        return (np.log(q) - special.log1p(-f))**(1./c)

    def _sf(self, x, c, q):
        return np.exp(-x**c + np.log(q))

    def _penalized_nnlf(self, theta, x):
        '''
        Return special purspose lsq objective error function to minimize.
        Method is overwritten to hook into scipy.stats optimization framework.

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
        f_ecdf = np.array([(i + 1 - 0.3)/(N + 0.4) for i in range(N)])
        error = np.abs(log(1/(1 - f_ecdf)) - 
                       log(1/(1 - self._cdf(x, *args))))**2.
        return np.sum(error) + Nbad * 10000.
genexpt = gen_exp_tail_gen(name='genexpt', a=0.)


class acer_o1_gen(rv_continuous):
    """
    ACER (1st order) extreme value distribution.
    """
    def _pdf(self, x, c, q):
        return c*x**(c-1.)*exp(-x**c + log(q))

    def _logpdf(self, x, c, q):
        return log(c) + (c-1)*log(x) - x**c + log(q)

    def _cdf(self, x, c, q):
        return -special.expm1(-x**c + log(q))

    def _logcdf(self, x, c, q):
        return special.log1p(-np.exp(-x**c + log(q)))

    def _ppf(self, f, c, q):
        return (np.log(q) - special.log1p(-f))**(1./c)

    def _sf(self, x, c, q):
        return np.exp(-x**c + np.log(q))

    def _penalized_nnlf(self, theta, x):
        '''
        Return special purspose lsq objective error function to minimize.
        Method is overwritten to hook into scipy.stats optimization framework.

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
        f_ecdf = np.array([(i + 1 - 0.3)/(N + 0.4) for i in range(N)])
        error = np.abs(log(1/(1 - f_ecdf)) - 
                       log(1/(1 - self._cdf(x, *args))))**2.
        return np.sum(error) + Nbad * 10000.
genexpt = gen_exp_tail_gen(name='genexpt', a=0.)
