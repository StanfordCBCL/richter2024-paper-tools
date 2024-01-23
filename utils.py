#!/usr/bin/env python
# coding=utf-8

import os

FDIR = os.path.dirname(__file__)

# define paths
f_geometries = os.path.join(FDIR, "geometries_vmr_richter22.txt")
f_geo_in = os.path.join(FDIR, "data", "geometric_pfaller22", "input")
f_cali_0d_in = os.path.join(FDIR, "data", "calibrated_richter24_from_0d", "input")
f_cali_0d_out = os.path.join(FDIR, "data", "calibrated_richter24_from_0d", "output")
f_cali_3d_in = os.path.join(FDIR, "data", "calibrated_richter24_from_0d", "input")
f_cali_3d_out = os.path.join(FDIR, "data", "calibrated_richter24_from_3d", "output")
