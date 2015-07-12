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

from colour.blindness import (
    CVD_MATRICES_MACHADO_2010,
    cvd_matrix_Machado2010,
    anomalous_trichromacy_cmfs_Machado2010,
    anomalous_trichromacy_matrix_Machado2010)
from colour.characterisation import DISPLAYS_RGB_PRIMARIES
from colour.colorimetry import LMS_CMFS
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestAnomalousTrichromacyCmfsMachado2010',
           'TestAnomalousTrichromacyMatrixMachado2010',
           'TestCvdMatrixMachado2010']


class TestAnomalousTrichromacyCmfsMachado2010(unittest.TestCase):
    """
    Defines
    :func:`colour.blindness.machado2010.anomalous_trichromacy_cmfs_Machado2010`
    definition unit tests methods.
    """

    def test_anomalous_trichromacy_cmfs_Machado2010(self):
        """
        Tests
        :func:`colour.blindness.machado2010.anomalous_trichromacy_cmfs_Machado2010`  #noqa
        definition.
        """

        cmfs = LMS_CMFS.get('Smith & Pokorny 1975 Normal Trichromats')
        np.testing.assert_almost_equal(
            anomalous_trichromacy_cmfs_Machado2010(
                cmfs,
                np.array([0, 0, 0]))[450],
            cmfs[450],
            decimal=7)

        np.testing.assert_almost_equal(
            anomalous_trichromacy_cmfs_Machado2010(
                cmfs,
                np.array([1, 0, 0]))[450],
            np.array([0.036317, 0.0635, 0.91]),
            decimal=7)

        np.testing.assert_almost_equal(
            anomalous_trichromacy_cmfs_Machado2010(
                cmfs,
                np.array([0, 1, 0]))[450],
            np.array([0.0343, 0.06178404, 0.91]),
            decimal=7)

        np.testing.assert_almost_equal(
            anomalous_trichromacy_cmfs_Machado2010(
                cmfs,
                np.array([0, 0, 1]))[450],
            np.array([0.0343, 0.0635, 0.9227024]),
            decimal=7)

        np.testing.assert_almost_equal(
            anomalous_trichromacy_cmfs_Machado2010(
                cmfs,
                np.array([10, 0, 0]))[450],
            np.array([0.05447001, 0.0635, 0.91]),
            decimal=7)

        np.testing.assert_almost_equal(
            anomalous_trichromacy_cmfs_Machado2010(
                cmfs,
                np.array([0, 10, 0]))[450],
            np.array([0.0343, 0.04634036, 0.91]),
            decimal=7)

        np.testing.assert_almost_equal(
            anomalous_trichromacy_cmfs_Machado2010(
                cmfs,
                np.array([0, 0, 10]))[450],
            np.array([0.0343, 0.0635, 1.]),
            decimal=7)


class TestAnomalousTrichromacyMatrixMachado2010(unittest.TestCase):
    """
    Defines
    :func:`colour.blindness.machado2010.anomalous_trichromacy_matrix_Machado2010`
    definition unit tests methods.
    """

    def test_anomalous_trichromacy_matrix_Machado2010(self):
        """
        Tests
        :func:`colour.blindness.machado2010.anomalous_trichromacy_matrix_Machado2010`  #noqa
        definition.
        """

        cmfs = LMS_CMFS.get('Smith & Pokorny 1975 Normal Trichromats')
        primaries = DISPLAYS_RGB_PRIMARIES['Typical CRT Brainard 1997']
        np.testing.assert_almost_equal(
            anomalous_trichromacy_matrix_Machado2010(
                cmfs,
                primaries,
                np.array([0, 0, 0])),
            np.identity(3),
            decimal=7)

        np.testing.assert_allclose(
            anomalous_trichromacy_matrix_Machado2010(
                cmfs,
                primaries,
                np.array([2, 0, 0])),
            CVD_MATRICES_MACHADO_2010.get('Protanomaly').get(0.1),
            rtol=0.0001,
            atol=0.0001)

        np.testing.assert_allclose(
            anomalous_trichromacy_matrix_Machado2010(
                cmfs,
                primaries,
                np.array([20, 0, 0])),
            CVD_MATRICES_MACHADO_2010.get('Protanomaly').get(1.0),
            rtol=0.0001,
            atol=0.0001)

        np.testing.assert_allclose(
            anomalous_trichromacy_matrix_Machado2010(
                cmfs,
                primaries,
                np.array([0, 2, 0])),
            CVD_MATRICES_MACHADO_2010.get('Deuteranomaly').get(0.1),
            rtol=0.0001,
            atol=0.0001)

        np.testing.assert_allclose(
            anomalous_trichromacy_matrix_Machado2010(
                cmfs,
                primaries,
                np.array([0, 20, 0])),
            CVD_MATRICES_MACHADO_2010.get('Deuteranomaly').get(1.0),
            rtol=0.0001,
            atol=0.0001)

        # TODO: Check inconsistency with ground truth values, d_S domain seems
        # to be [5, 59] instead of [0, 20].
        np.testing.assert_allclose(
            anomalous_trichromacy_matrix_Machado2010(
                cmfs,
                primaries,
                np.array([0, 0, 5])),
            CVD_MATRICES_MACHADO_2010.get('Tritanomaly').get(0.1),
            rtol=0.0001,
            atol=0.0001)

        np.testing.assert_allclose(
            anomalous_trichromacy_matrix_Machado2010(
                cmfs,
                primaries,
                np.array([0, 0, 59])),
            CVD_MATRICES_MACHADO_2010.get('Tritanomaly').get(1.0),
            rtol=0.001,
            atol=0.001)


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
