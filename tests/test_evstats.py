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