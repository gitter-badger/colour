#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation of CVD - Machado (2010)
==================================

Defines Machado (2010) objects for simulation of colour vision deficiency:

-   :func:`anomalous_trichromacy_cmfs_Machado2010`
-   :func:`anomalous_trichromacy_matrix_Machado2010`
-   :func:`cvd_matrix_Machado2010`

See Also
--------
`Machado (2010) - CVD IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/blob/master/notebooks/cvd/machado2010.ipynb>`_  # noqa

References
----------
.. [1]  Machado, G. (2010). A model for simulation of color vision deficiency
        and a color contrast enhancement technique for dichromats, (September).
        Retrieved from http://www.lume.ufrgs.br/handle/10183/26950
.. [2]  Protanopia – Red-Green Color Blindness. (n.d.). Retrieved July 4, 2015,
        from
        http://www.color-blindness.com/protanopia-red-green-color-blindness/
.. [3]  Deuteranopia – Red-Green Color Blindness. (n.d.). Retrieved July 4,
        2015, from
        http://www.color-blindness.com/deuteranopia-red-green-color-blindness/
.. [4]  Tritanopia – Blue-Yellow Color Blindness. (n.d.). Retrieved July 4,
        2015, from
        http://www.color-blindness.com/tritanopia-blue-yellow-color-blindness/
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.blindness import CVD_MATRICES_MACHADO_2010
from colour.colorimetry import LMS_CMFS, SpectralShape
from colour.utilities import dot_matrix, dot_vector, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LMS_TO_WSYBRG_MATRIX',
           'RGB_to_WSYBRG_matrix',
           'anomalous_trichromacy_cmfs_Machado2010',
           'anomalous_trichromacy_matrix_Machado2010',
           'cvd_matrix_Machado2010']

LMS_TO_WSYBRG_MATRIX = np.array(
    [[0.600, 0.400, 0.000],
     [0.240, 0.105, -0.700],
     [1.200, -1.600, 0.400]])
"""
Ingling and Tsou (1977) matrix converting from cones responses to
opponent-colour space.

LMS_TO_WSYBRG_MATRIX : array_like, (3, 3)
"""


def RGB_to_WSYBRG_matrix(cmfs, primaries):
    wavelengths = cmfs.wavelengths
    WSYBRG = dot_vector(LMS_TO_WSYBRG_MATRIX, cmfs.values)
    WS, YB, RG = tsplit(WSYBRG)

    primaries = primaries.clone().align(cmfs.shape, left=0, right=0)
    R, G, B = tsplit(primaries.values)

    WS_R = np.trapz(R * WS, wavelengths)
    WS_G = np.trapz(G * WS, wavelengths)
    WS_B = np.trapz(B * WS, wavelengths)

    YB_R = np.trapz(R * YB, wavelengths)
    YB_G = np.trapz(G * YB, wavelengths)
    YB_B = np.trapz(B * YB, wavelengths)

    RG_R = np.trapz(R * RG, wavelengths)
    RG_G = np.trapz(G * RG, wavelengths)
    RG_B = np.trapz(B * RG, wavelengths)

    M_G = np.array([[WS_R, WS_G, WS_B],
                    [YB_R, YB_G, YB_B],
                    [RG_R, RG_G, RG_B]])

    PWS = 1 / (WS_R + WS_G + WS_B)
    PYB = 1 / (YB_R + YB_G + YB_B)
    PRG = 1 / (RG_R + RG_G + RG_B)

    M_G *= np.array([PWS, PYB, PRG])[:, np.newaxis]

    return M_G


def anomalous_trichromacy_cmfs_Machado2010(cmfs, d_LMS):
    """
    Shifts given *LMS* cone fundamentals colour matching functions with given
    :math:`\Delta_{LMS}` shift amount in nanometers to simulate anomalous
    trichromacy.

    Parameters
    ----------
    cmfs : LMS_ConeFundamentals
        *LMS* cone fundamentals colour matching functions.
    d_LMS : array_like
        :math:`\Delta_{LMS}` shift amount in nanometers as an array of ints.

    Notes
    -----
    -   Input *LMS* cone fundamentals colour matching functions steps size is
        expected to be 1 nanometer, non complying input will be interpolated
        at 1 nanometer steps size.
    -   Input :math:`\Delta_{LMS}` shift amount is in domain [0, 20].

    Returns
    -------
    LMS_ConeFundamentals
        Anomalous trichromacy *LMS* cone fundamentals colour matching
        functions.

    Examples
    --------
    >>> cmfs = LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals')
    >>> cmfs[450]
    array([ 0.0498639,  0.0870524,  0.955393 ])
    >>> anomalous_trichromacy_cmfs_Machado2010(cmfs, np.array([15, 0, 0]))[450]  #noqa  # doctest: +ELLIPSIS
    array([ 0.0891288...,  0.0870524 ,  0.955393  ])
    """

    cmfs = cmfs.clone()
    if cmfs.shape.steps != 1:
        cmfs = cmfs.clone().interpolate(SpectralShape(steps=1))

    L, M, S = tsplit(cmfs.values)
    d_L, d_M, d_S = np.asarray(d_LMS).astype(np.int_)

    wavelengths = cmfs.wavelengths
    area_L = np.trapz(L, wavelengths)
    area_M = np.trapz(M, wavelengths)

    alpha = lambda x: (20 - x) / 20

    # Corrected equations as per:
    # http://www.inf.ufrgs.br/~oliveira/pubs_files/
    # CVD_Simulation/CVD_Simulation.html#Errata
    L_a = alpha(d_L) * L + 0.96 * area_L / area_M * (1 - alpha(d_L)) * M
    M_a = alpha(d_M) * M + 1 / 0.96 * area_M / area_L * (1 - alpha(d_M)) * L
    # TODO: Check inconsistency with ground truth values, d_S domain seems
    # to be [5, 59] instead of [0, 20].
    S_a = cmfs.s_bar.clone().shift(d_S).values

    LMS_a = tstack((L_a, M_a, S_a))
    cmfs[wavelengths] = LMS_a

    severity = '{0}, {1}, {2}'.format(d_L, d_M, d_S)
    template = '{0} - Anomalous trichromacy ({1})'
    cmfs.name = template.format(cmfs.name, severity)
    cmfs.title = template.format(cmfs.title, severity)

    return cmfs


def anomalous_trichromacy_matrix_Machado2010(cmfs, primaries, d_LMS):
    """
    Computes Machado (2010) *CVD* matrix for given *LMS* cone fundamentals
    colour matching functions and display primaries tri-spectral power
    distributions with given :math:`\Delta_{LMS}` shift amount in nanometers to
    simulate anomalous trichromacy.

    Parameters
    ----------
    cmfs : LMS_ConeFundamentals
        *LMS* cone fundamentals colour matching functions.
    primaries : RGB_DisplayPrimaries
        *RGB* display primaries tri-spectral power distributions.
    d_LMS : array_like
        :math:`\Delta_{LMS}` shift amount in nanometers as an array of ints.

    Notes
    -----
    -   Input *LMS* cone fundamentals colour matching functions steps size is
        expected to be 1 nanometer, non complying input will be interpolated
        at 1 nanometer steps size.
    -   Input :math:`\Delta_{LMS}` shift amount is in domain [0, 20].

    Returns
    -------
    ndarray
        Anomalous trichromacy matrix.

    Examples
    --------
    >>> cmfs = LMS_CMFS.get('Stockman & Sharpe 2 Degree Cone Fundamentals')
    >>> cmfs[450]
    array([ 0.0498639,  0.0870524,  0.955393 ])
    >>> anomalous_trichromacy_cmfs_Machado2010(cmfs, np.array([15, 0, 0]))[450]  #noqa  # doctest: +ELLIPSIS
    array([ 0.0891288...,  0.0870524 ,  0.955393  ])
    """

    if cmfs.shape.steps != 1:
        cmfs = cmfs.clone().interpolate(SpectralShape(steps=1))

    M_n = RGB_to_WSYBRG_matrix(cmfs, primaries)
    cmfs_a = anomalous_trichromacy_cmfs_Machado2010(cmfs, d_LMS)
    M_a = RGB_to_WSYBRG_matrix(cmfs_a, primaries)

    return dot_matrix(np.linalg.inv(M_n), M_a)


def cvd_matrix_Machado2010(deficiency, severity):
    """
    Computes Machado (2010) *CVD* matrix for given deficiency and severity
    using the pre-computed matrices dataset.

    Parameters
    ----------
    deficiency : unicode
        {'Protanomaly', 'Deuteranomaly', 'Tritanomaly'}
        Colour blindness / vision deficiency types :
        - *Protanomaly* : defective long-wavelength cones (L-cones). The
        complete absence of L-cones is called *Protanopia* or *red-dichromacy*.
        - *Deuteranomaly* : defective medium-wavelength cones (M-cones) with
        peak of sensitivity being moved towards the red sensitive cones. The
        complete absence of M-cones is called *Deuteranopia*.
        - *Tritanomaly* : defective short-wavelength cones (S-cones), an
        alleviated form of blue-yellow color blindness. The complete absence of
        S-cones is called *Tritanopia*.
    severity : numeric
        Severity of the colour vision deficiency in domain [0, 1].

    Returns
    -------
    ndarray
        *CVD* matrix.

    Notes
    -----
    -   Input severity is in domain [0, 1].

    Examples
    --------
    >>> cvd_matrix_Machado2010('Protanomaly', 0.15)
    array([[ 0.7869875,  0.2694875, -0.0564735],
           [ 0.0431695,  0.933774 ,  0.023058 ],
           [-0.004238 , -0.0024515,  1.0066895]])
    """

    matrices = CVD_MATRICES_MACHADO_2010[deficiency]
    samples = np.array(sorted(matrices.keys()))
    index = min(np.searchsorted(samples, severity), len(samples) - 1)

    a = samples[index]
    b = samples[min(index + 1, len(samples) - 1)]

    m1, m2 = matrices[a], matrices[b]

    if a == b:
        # 1.0 severity CVD matrix, returning directly.
        return m1
    else:
        return m1 + (severity - a) * ((m2 - m1) / (b - a))
