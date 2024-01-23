#!/usr/bin/env python
# coding=utf-8

import os

FDIR = os.path.dirname(__file__)

# define paths
f_geometries = os.path.join(FDIR, "..", "geometries_vmr_richter22.txt")
f_geo_in = os.path.join(FDIR, "vmr", "geometric_pfaller22", "input")
f_cali_in = os.path.join(FDIR, "vmr", "calibrated_richter24_from_0d", "input")
f_cali_out = os.path.join(FDIR, "vmr", "calibrated_richter24_from_0d", "output")
