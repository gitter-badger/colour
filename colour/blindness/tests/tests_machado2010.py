# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.blindness.machado2010` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.blindness import cvd_matrix_Machado2010
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestCvdMatrixMachado2010']


class TestCvdMatrixMachado2010(unittest.TestCase):
    """
    Defines :func:`colour.blindness.machado2010.cvd_matrix_Machado2010`
    definition unit tests methods.
    """

    def test_cvd_matrix_Machado2010(self):
        """
        Tests :func:`colour.blindness.machado2010.cvd_matrix_Machado2010`
        definition.
        """

        np.testing.assert_almost_equal(
            cvd_matrix_Machado2010('Protanomaly', 0.0),
            np.array([[1., 0., -0.],
                      [0., 1., 0.],
                      [-0., -0., 1.]]),
            decimal=7)

        np.testing.assert_almost_equal(
            cvd_matrix_Machado2010('Deuteranomaly', 0.1),
            np.array([[0.866435, 0.177704, -0.044139],
                      [0.049567, 0.939063, 0.01137],
                      [-0.003453, 0.007233, 0.99622]]),
            decimal=7)

        np.testing.assert_almost_equal(
            cvd_matrix_Machado2010('Tritanomaly', 1.0),
            np.array([[1.255528, -0.076749, -0.178779],
                      [-0.078411, 0.930809, 0.147602],
                      [0.004733, 0.691367, 0.3039]]),
            decimal=7)

        np.testing.assert_almost_equal(
            cvd_matrix_Machado2010('Tritanomaly', 0.55),
            np.array([[1.060887, -0.0150435, -0.0458435],
                      [-0.0189575, 0.9677475, 0.0512115],
                      [0.003177, 0.275137, 0.721686]]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_cvd_matrix_Machado2010(self):
        """
        Tests :func:`colour.blindness.machado2010.cvd_matrix_Machado2010`
        definition nan support.
        """

        for case in [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]:
            cvd_matrix_Machado2010('Tritanomaly', case)


if __name__ == '__main__':
    unittest.main()
