import unittest

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

    def test_instance_weibull_min(self):
        calculated = dist.weibull_min.cdf(2.5, 2., loc=0.5, scale=2.)
        expected = self.dist.cdf(2.5, 2., loc=0.5, scale=2.)
        self.assertAlmostEqual(calculated, expected, places=4)


class Test_gumbel_r_gen(unittest.TestCase):
    def setUp(self):
        self.dist = dist._distns.gumbel_r_gen()

    def tearDown(self):
        pass

    def test_cdf(self):
        calculated = self.dist.cdf(3., loc=1., scale=2.)
        expected = np.exp(-np.exp(-1.))
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_pdf(self):
        calculated = self.dist.pdf(3., loc=1., scale=2.)
        expected = np.exp(-1. - np.exp(-1.))/2.
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_ppf(self):
        calculated = self.dist.ppf(0.5, loc=1., scale=2.)
        expected = 1. - 2.*np.log(np.log(2))
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_instance_gumbel(self):
        calculated = dist.gumbel.cdf(2.5, loc=0.5, scale=2.)
        expected = self.dist.cdf(2.5, loc=0.5, scale=2.)
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_instance_gumbel_max(self):
        calculated = dist.gumbel_max.cdf(2.5, loc=0.5, scale=2.)
        expected = self.dist.cdf(2.5, loc=0.5, scale=2.)
        self.assertAlmostEqual(calculated, expected, places=4)


class Test_gen_exp_tail_gen(unittest.TestCase):
    def setUp(self):
        self.dist = dist._distns.gen_exp_tail_gen()

    def tearDown(self):
        pass

    def test_cdf(self):
        calculated = self.dist.cdf(2.5, 2., 1., loc=0.5, scale=2.)
        expected = -np.expm1(-1.)
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_pdf(self):
        calculated = self.dist.pdf(2.5, 2., 1., loc=0.5, scale=2.)
        expected = np.exp(-1.)
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_ppf(self):
        calculated = self.dist.ppf(0.5, 2., 1., loc=0.5, scale=2.)
        expected = 2.*np.sqrt(np.log(2.)) + 0.5
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_instance_gen_exp_tail(self):
        calculated = dist.genexptail.cdf(2.5, 2., 1., loc=0.5, scale=2.)
        expected = self.dist.cdf(2.5, 2., 1., loc=0.5, scale=2.)
        self.assertAlmostEqual(calculated, expected, places=4)


class Test_acer_o1_gen(unittest.TestCase):
    def setUp(self):
        self.dist = dist._distns.acer_o1_gen()

    def tearDown(self):
        pass

    def test_cdf(self):
        calculated = self.dist.cdf(3., 1., 1., loc=1., scale=2.)
        expected = np.exp(-np.exp(-1.))
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_pdf(self):
        calculated = self.dist.pdf(3., 1., 1., loc=1., scale=2.)
        expected = np.exp(-1. - np.exp(-1.))/2.
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_ppf(self):
        calculated = self.dist.ppf(0.5, 1., 1., loc=1., scale=2.)
        expected = 1. - 2.*np.log(np.log(2))
        self.assertAlmostEqual(calculated, expected, places=4)

    def test_instance_acer_o1(self):
        calculated = dist.acer_o1.cdf(2.5, 1., 1., loc=0.5, scale=2.)
        expected = self.dist.cdf(2.5, 1., 1., loc=0.5, scale=2.)
        self.assertAlmostEqual(calculated, expected, places=4)
