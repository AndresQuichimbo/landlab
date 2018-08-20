#!/usr/env/python

import pytest
import numpy as np

from landlab.components import LakeMapperBarnes
from landlab import RasterModelGrid, HexModelGrid

"""
These tests test specific aspects of LakeMapperBarnes not picked up in the
various docstrings.
"""
rmg = RasterModelGrid((5, 5), dx=2.)
hmg = HexModelGrid(30, 29 dx=3.)
rmg.add_zeros('node', 'topographic__elevation', dtype=float)
hmg.add_zeros('node', 'topographic__elevation', dtype=float)


def test_bad_init_method1(rmg):
    with pytest.raises(ValueError):
        lmb = LakeMapperBarnes(rmg, method='Nope')


def test_bad_init_method1(rmg):
    with pytest.raises(ValueError):
        lmb = LakeMapperBarnes(rmg, method='d8')


def test_bad_init_gridmethod(rmg):
    with pytest.raises(ValueError):
        lmb = LakeMapperBarnes(hmg, method='D8')