'''
Notes
-----

The distributions will rely on the `scipy.stats` framework. This may
imply that some functions can rely on private functions in the scipy
package.

Similar to scipy.stats._continous_distns.

Thanks to all SciPy developers for their great work!
'''


from __future__ import division

from scipy import special
from scipy.stats import rv_continuous

import numpy as np

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
#    def _rvs(self):
#        return chi.rvs(2, size=self._size, random_state=self._random_state)

    def _pdf(self, r):
        return r * np.exp(-0.5 * r**2)

    def _cdf(self, r):
        return -special.expm1(-0.5 * r**2)

    def _ppf(self, q):
        return np.sqrt(-2 * special.log1p(-q))

    def _sf(self, r):
        return np.exp(-0.5 * r**2)

    def _isf(self, q):
        return np.sqrt(-2 * log(q))

#    def _stats(self):
#        val = 4 - pi
#        return (np.sqrt(pi/2), val/2, 2*(pi-3)*sqrt(pi)/val**1.5,
#                6*pi/val-16/val**2)

#    def _entropy(self):
#        return _EULER/2.0 + 1 - 0.5*log(2)
rayleigh = rayleigh_gen(a=0.0, name="rayleigh")
