'''
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

from ._optimize import _residual_error


#  Special constants
_EULER = 0.577215664901532860606512090082402431042
_ZETA3 = 1.202056903159594285399738161511449990765


class rayleigh_gen(rv_continuous):
    """A Rayleigh continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `rayleigh` is::

        rayleigh.pdf(x) = x * exp(-x**2/2)

    for ``x >= 0``.

    %(after_notes)s

    %(example)s

    """
    def _pdf(self, x):
        return x*exp(-0.5*x**2)

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
    """
    A Frechet right class continuous random variable.

    %(before_notes)s

    See Also
    --------
    weibull, weibull_min

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
    """
    A right-skewed Gumbel continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel, gumbel_max

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
    A generalized exponential tail continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `gen_exp_tail` is::

        gen_exp_tail.pdf(x, c, q) = c*x**(c-1.)*exp(-x**c + log(q))

    The generelized exponential tail distribution is related to 
    peak-over-threshold distributions. It assumes a exponentially decreasing
    tail and is in the domain of attraction of type I extreme value 
    distribution.

    The MLE for this distribution do not exist. Thus, ``fit`` method is 
    overwritten with a least-square procedure. It is also recommended to lock
    the location parameter for more robust parameter estimates.

    %(after_notes)s

    %(example)s

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
        Method is overwritten to hook into scipy.stats optimization framework
        with custom residual error function. ML estimator do not exist.

        '''
        def y_fun(cdf):
            return log(1/(1 - cdf))

        return _residual_error(self, theta, x, y_fun)
genexptail = gen_exp_tail_gen(name='genexptail', a=0.)


class acer_o1_gen(rv_continuous):
    """
    A generalized Gumbel-like continuous random variable.

    %(before_notes)s

    See Also
    --------
    gumbel, gumbel_max

    Notes
    -----
    The probability density function for `acer_o1` is::

        acer_o1.pdf(x, c, qn) = c*qn*x^(c-1)*exp(-exp(-x**c + log(qn))-x**c)

    The generelized Gumbel-like distributions. It assumes a exponentially 
    decreasing tail and will approach a type I (Gumbel) extreme value 
    distribution. Note that for c=1 and qn=1 the distribution reduces to the
    conventional Gumbel distribution.

    The MLE for this distribution do not exist. Thus, ``fit`` method is 
    overwritten with a least-square procedure. It is also recommended to lock
    the location parameter for more robust parameter estimates.

    %(after_notes)s

    References
    ----------
    [1]. Naess et al.

    %(example)s

    """
    def _pdf(self, x, c, qn):
        return c*qn*x**(c-1)*exp(-exp(-x**c + log(qn))-x**c)

    def _logpdf(self, x, c, q):
        return log(c*q*x**(c-1)) + (-exp(-x**c + log(qn))-x**c)

    def _cdf(self, x, c, qn):
        return exp(-exp(-x**c + log(qn)))

    def _logcdf(self, x, c, qn):
        return -exp(-x**c + log(qn))

    def _ppf(self, f, c, qn):
        return (-(log(-(log(f))/qn)))**(1/c)

    def _penalized_nnlf(self, theta, x):
        '''
        Method is overwritten to hook into scipy.stats optimization framework
        with custom residual error function. ML estimator do not exist.

        '''
        def y_fun(cdf):
            return log(-1./log(cdf))

        return _residual_error(self, theta, x, y_fun)
acer_o1 = acer_o1_gen(name='acer_o1')
