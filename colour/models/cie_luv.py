#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CIE Luv Colourspace
===================

Defines the *CIE Luv* colourspace transformations:

-   :func:`XYZ_to_Luv`
-   :func:`Luv_to_XYZ`
-   :func:`Luv_to_uv`
-   :func:`Luv_uv_to_xy`
-   :func:`Luv_to_LCHuv`
-   :func:`LCHuv_to_Luv`

See Also
--------
`CIE Luv Colourspace IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/cie_luv.ipynb>`_

References
----------
.. [1]  Wikipedia. (n.d.). CIELUV. Retrieved February 24, 2014, from
        http://en.wikipedia.org/wiki/CIELUV
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.constants import CIE_E, CIE_K
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_Luv',
           'Luv_to_XYZ',
           'Luv_to_uv',
           'Luv_uv_to_xy',
           'Luv_to_LCHuv',
           'LCHuv_to_Luv']


def XYZ_to_Luv(XYZ,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE XYZ* tristimulus values to *CIE Luv* colourspace.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    illuminant : array_like, optional
        Reference *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    ndarray
        *CIE Luv* colourspace array.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 1].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [2]  Lindbloom, B. (2003). XYZ to Luv. Retrieved February 24, 2014,
            from http://brucelindbloom.com/Eqn_XYZ_to_Luv.html

    Examples
    --------
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
    >>> XYZ_to_Luv(XYZ)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -28.7922944...,  -1.3558195...])
    """

    X, Y, Z = tsplit(XYZ)
    X_r, Y_r, Z_r = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    y_r = Y / Y_r

    L = np.where(y_r > CIE_E, 116 * y_r ** (1 / 3) - 16, CIE_K * y_r)

    u = (13 * L * ((4 * X / (X + 15 * Y + 3 * Z)) -
                   (4 * X_r / (X_r + 15 * Y_r + 3 * Z_r))))
    v = (13 * L * ((9 * Y / (X + 15 * Y + 3 * Z)) -
                   (9 * Y_r / (X_r + 15 * Y_r + 3 * Z_r))))

    Luv = tstack((L, u, v))

    return Luv


def Luv_to_XYZ(Luv,
               illuminant=ILLUMINANTS.get(
                   'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Converts from *CIE Luv* colourspace to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    Luv : array_like
        *CIE Luv* colourspace array.
    illuminant : array_like, optional
        Reference *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input :math:`L^*` is in domain [0, 100].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output *CIE XYZ* tristimulus values are in domain [0, 1].

    References
    ----------
    .. [3]  Lindbloom, B. (2003). Luv to XYZ. Retrieved February 24, 2014,
            from http://brucelindbloom.com/Eqn_Luv_to_XYZ.html

    Examples
    --------
    >>> Luv = np.array([37.98562910, -28.79229446, -1.35581950])
    >>> Luv_to_XYZ(Luv)  # doctest: +ELLIPSIS
    array([ 0.0704953...,  0.1008    ,  0.0955831...])
    """

    L, u, v = tsplit(Luv)
    X_r, Y_r, Z_r = tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    Y = np.where(L > CIE_E * CIE_K, ((L + 16) / 116) ** 3, L / CIE_K)

    a = 1 / 3 * ((52 * L / (u + 13 * L *
                            (4 * X_r / (X_r + 15 * Y_r + 3 * Z_r)))) - 1)
    b = -5 * Y
    c = -1 / 3.0
    d = Y * (39 * L / (v + 13 * L *
                       (9 * Y_r / (X_r + 15 * Y_r + 3 * Z_r))) - 5)

    X = (d - b) / (a - c)
    Z = X * a + b

    XYZ = tstack((X, Y, Z))

    return XYZ


def Luv_to_uv(Luv,
              illuminant=ILLUMINANTS.get(
                  'CIE 1931 2 Degree Standard Observer').get('D50')):
    """
    Returns the :math:`uv^p` chromaticity coordinates from given *CIE Luv*
    colourspace array.

    Parameters
    ----------
    Luv : array_like
        *CIE Luv* colourspace array.
    illuminant : array_like, optional
        Reference *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array.

    Returns
    -------
    ndarray
        :math:`uv^p` chromaticity coordinates.

    Notes
    -----
    -   Input :math:`L^*` is in domain [0, 100].
    -   Input *illuminant* *xy* chromaticity coordinates or *CIE xyY*
        colourspace array are in domain [0, :math:`\infty`].
    -   Output :math:`uv^p` chromaticity coordinates are in domain [0, 1].

    References
    ----------
    .. [4]  Wikipedia. (n.d.). The forward transformation. Retrieved February
            24, 2014, from
            http://en.wikipedia.org/wiki/CIELUV#The_forward_transformation

    Examples
    --------
    >>> Luv = np.array([37.98562910, -28.79229446, -1.35581950])
    >>> Luv_to_uv(Luv)  # doctest: +ELLIPSIS
    array([ 0.1508531...,  0.4853297...])
    """

    X, Y, Z = tsplit(Luv_to_XYZ(Luv, illuminant))

    uv = tstack((4 * X / (X + 15 * Y + 3 * Z),
                 9 * Y / (X + 15 * Y + 3 * Z)))

    return uv


def Luv_uv_to_xy(uv):
    """
    Returns the *xy* chromaticity coordinates from given *CIE Luv* colourspace
    :math:`uv^p` chromaticity coordinates.

    Parameters
    ----------
    uv : array_like
        *CIE Luv u"v"* chromaticity coordinates.

    Returns
    -------
    ndarray
        *xy* chromaticity coordinates.

    Notes
    -----
    -   Input :math:`uv^p` chromaticity coordinates are in domain [0, 1].
    -   Output *xy* is in domain [0, 1].

    References
    ----------
    .. [5]  Wikipedia. (n.d.). The reverse transformation. Retrieved from
            http://en.wikipedia.org/wiki/CIELUV#The_reverse_transformation

    Examples
    --------
    >>> uv = np.array([0.15085309882985695, 0.48532970854318019])
    >>> Luv_uv_to_xy(uv)  # doctest: +ELLIPSIS
    array([ 0.2641477...,  0.3777000...])
    """

    u, v = tsplit(uv)

    xy = tstack((9 * u / (6 * u - 16 * v + 12),
                 4 * v / (6 * u - 16 * v + 12)))

    return xy


def Luv_to_LCHuv(Luv):
    """
    Converts from *CIE Luv* colourspace to *CIE LCHuv* colourspace.

    Parameters
    ----------
    Luv : array_like
        *CIE Luv* colourspace array.

    Returns
    -------
    ndarray
        *CIE LCHuv* colourspace array.

    Notes
    -----
    -   :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [6]  Lindbloom, B. (2003). Luv to LCH(uv). Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_Luv_to_LCH.html

    Examples
    --------
    >>> Luv = np.array([37.98562910, -28.79229446, -1.35581950])
    >>> Luv_to_LCHuv(Luv)  # doctest: +ELLIPSIS
    array([  37.9856291...,   28.8241993...,  182.6960474...])
    """

    L, u, v = tsplit(Luv)

    H = np.array(180 * np.arctan2(v, u) / np.pi)
    H[np.array(H < 0)] += 360

    LCHuv = tstack((L, np.sqrt(u ** 2 + v ** 2), H))

    return LCHuv


def LCHuv_to_Luv(LCHuv):
    """
    Converts from *CIE LCHuv* colourspace to *CIE Luv* colourspace.

    Parameters
    ----------
    LCHuv : array_like
        *CIE LCHuv* colourspace array.

    Returns
    -------
    ndarray
        *CIE Luv* colourspace array.

    Notes
    -----
    -   :math:`L^*` is in domain [0, 100].

    References
    ----------
    .. [7]  Lindbloom, B. (2006). LCH(uv) to Luv. Retrieved February 24, 2014,
            from http://www.brucelindbloom.com/Eqn_LCH_to_Luv.html

    Examples
    --------
    >>> LCHuv = np.array([37.98562910, 28.82419933, 182.69604747])
    >>> LCHuv_to_Luv(LCHuv)  # doctest: +ELLIPSIS
    array([ 37.9856291..., -28.7922944...,  -1.3558195...])
    """

    L, C, H = tsplit(LCHuv)

    Luv = tstack((L, C * np.cos(np.radians(H)), C * np.sin(np.radians(H))))

    return Luv
