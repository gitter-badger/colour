#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation of CVD - Machado (2010)
=================================

Defines Machado (2010) objects for simulation of colour vision deficiency:

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

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['cvd_matrix_Machado2010']


def cvd_matrix_Machado2010(deficiency, severity):
    """
    Computes Machado (2010) *CVD* matrix for given deficiency and severity.

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
