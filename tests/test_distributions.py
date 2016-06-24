import unittest
from mock import Mock, patch

import numpy as np


import evapy.distributions as dist


class Test_rayleigh_gen(unittest.TestCase):
    def setUp(self):
        self.dist = dist._distns.rayleigh_gen()

    def tearDown(self):
        pass

    def test_cdf(self):
        calculated = self.dist.cdf(2.5, loc=0.5, scale=np.sqrt(2.))
        expected = -np.expm1(-1.)
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_pdf(self):
        calculated = self.dist.pdf(2.5, loc=0.5, scale=np.sqrt(2.))
        expected = np.exp(-1.)
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_ppf(self):
        calculated = self.dist.ppf(0.5, loc=0.5, scale=np.sqrt(2.))
        expected = 2.*np.sqrt(np.log(2.)) + 0.5
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_instance_rayleigh(self):
        calculated = dist.rayleigh.cdf(2.5, loc=0.5, scale=np.sqrt(2.))
        expected = -np.expm1(-1.)
        self.assertAlmostEqual(calculated, expected, places=4)


class Test_frechet_r_gen(unittest.TestCase):
    def setUp(self):
        self.dist = dist._distns.frechet_r_gen()

    def tearDown(self):
        pass

    def test_cdf(self):
        calculated = self.dist.cdf(2.5, 2., loc=0.5, scale=2.)
        expected = -np.expm1(-1.)
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_pdf(self):
        calculated = self.dist.pdf(2.5, 2., loc=0.5, scale=2.)
        expected = np.exp(-1.)
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_ppf(self):
        calculated = self.dist.ppf(0.5, 2., loc=0.5, scale=2.)
        expected = 2.*np.sqrt(np.log(2.)) + 0.5
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_instance_weibull(self):
        calculated = dist.weibull.cdf(2.5, 2., loc=0.5, scale=2.)
        expected = self.dist.cdf(2.5, 2., loc=0.5, scale=2.)
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_instance_weibull_max(self):
        calculated = dist.weibull_min.cdf(2.5, 2., loc=0.5, scale=2.)
        expected = self.dist.cdf(2.5, 2., loc=0.5, scale=2.)
        self.assertAlmostEqual(calculated, expected, places=4)