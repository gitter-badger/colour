#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .dataset import *  # noqa
from . import dataset
from .machado2010 import (
    anomalous_trichromacy_cmfs_Machado2010,
    anomalous_trichromacy_matrix_Machado2010,
    cvd_matrix_Machado2010)

__all__ = []
__all__ += dataset.__all__
__all__ += ['anomalous_trichromacy_cmfs_Machado2010',
            'anomalous_trichromacy_matrix_Machado2010',
            'cvd_matrix_Machado2010']
