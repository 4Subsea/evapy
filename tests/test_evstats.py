import unittest
from mock import Mock, patch

import numpy as np

from evapy import evstats


class Test__argrelmax(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple_find(self):
        x = np.array([0., 1., 0., -1., 0., 2.])
        calculated = evstats._argrelmax(x)
        expected = np.array([False, True, False, False, False, False])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_none(self):
        x = np.array([0., 1., 2., 3., 4., 5.])
        calculated = evstats._argrelmax(x)
        expected = np.array([False, False, False, False, False, False])
        np.testing.assert_array_equal(calculated, expected)


class Test__argupcross(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple_find(self):
        x = np.array([0., 1., -1., -2., -1., 1., 0.])
        calculated = evstats._argupcross(x, x_up=0.)
        expected = np.array([True, False, False, False, True, False, False])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_w_zero(self):
        x = np.array([0., 1., 0., -1., -2., -1., 0., 1., 0.])
        calculated = evstats._argupcross(x, x_up=0.)
        expected = np.array([True, False, False, False, False, False, True,
                             False, False])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_w_other(self):
        x = np.array([0., 1., 0., -1., -2., -1., 0., 1., 0.]) + 2.
        calculated = evstats._argupcross(x, x_up=2.)
        expected = np.array([True, False, False, False, False, False, True,
                             False, False])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_none(self):
        x = np.array([0.1, 1., 2., 3., 4., 5.])
        calculated = evstats._argupcross(x, x_up=0.)
        expected = np.array([False, False, False, False, False, False])
        np.testing.assert_array_equal(calculated, expected)


class Test_argrelmax(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple_find(self):
        x = np.array([0., 1., 0., -1., 0., 2.])
        calculated = evstats.argrelmax(x)
        expected = np.array([1])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_none(self):
        x = np.array([0., 1., 2., 3., 4., 5.])
        calculated = evstats.argrelmax(x)
        expected = np.array([5])
        np.testing.assert_array_equal(calculated, expected)


class Test_argupcross(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple_find(self):
        x = np.array([0., 1., -1., -2., -1., 1., 0.])
        calculated = evstats.argupcross(x, x_up=0.)
        expected = np.array([0., 4])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_w_zero(self):
        x = np.array([0., 1., 0., -1., -2., -1., 0., 1., 0.])
        calculated = evstats.argupcross(x, x_up=0.)
        expected = np.array([0, 6])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_w_other(self):
        x = np.array([0., 1., 0., -1., -2., -1., 0., 1., 0.]) + 2.
        calculated = evstats.argupcross(x, x_up=2.)
        expected = np.array([0., 6.])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_none(self):
        x = np.array([0.1, 1., 2., 3., 4., 5.])
        calculated = evstats.argupcross(x, x_up=0.)
        expected = np.array([0.])
        np.testing.assert_array_equal(calculated, expected)


class Test_argrelmax_decluster(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple_find(self):
        x = np.array([0., 1., 0., -1., 0., 2.])
        calculated = evstats.argrelmax_decluster(x, x_up=0.)
        expected = np.array([1])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_decluster(self):
        x = np.array([0., 1., 2., 1., 2., 3., 2., 1., 0., 2.,-1.,1.])
        calculated = evstats.argrelmax_decluster(x, x_up=0.)
        expected = np.array([5, 9.])
        np.testing.assert_array_equal(calculated, expected)

    def test_simple_find_none(self):
        x = np.array([0.1, 1., 2., 3., 4., 5.])
        calculated = evstats.argrelmax_decluster(x)
        expected = np.array([5])
        np.testing.assert_array_equal(calculated, expected)

    def test_find_decluster_below_upcross(self):
        x = np.array([0., 1., 2., 1., 2., 3., 2., 1., 0.,-1.,-2., -1, -2 , -1,
                      0., 1, 0., -1.,-2., -1., 0., 1.])
        calculated = evstats.argrelmax_decluster(x, x_up=0.)
        expected = np.array([5, 15.])
        np.testing.assert_array_equal(calculated, expected)

