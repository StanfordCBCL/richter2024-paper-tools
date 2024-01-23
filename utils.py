#!/usr/bin/env python
# coding=utf-8

import os

FDIR = os.path.dirname(__file__)

# list of geometries used in pfaller22 and richter24
f_geometries = os.path.join(FDIR, "geometries_vmr_pfaller22.txt")

# geometric 0d models and simulation results
f_geo_in = os.path.join(FDIR, "data", "geometric_pfaller22", "input")
f_geo_out = os.path.join(FDIR, "data", "geometric_pfaller22", "output")

# calibrated 0d models from 0d data and 3d data
f_cali_0d_in = os.path.join(FDIR, "data", "calibrated_richter24_from_0d", "input")
f_cali_0d_out = os.path.join(FDIR, "data", "calibrated_richter24_from_0d", "output")
f_cali_3d_in = os.path.join(FDIR, "data", "calibrated_richter24_from_0d", "input")
f_cali_3d_out = os.path.join(FDIR, "data", "calibrated_richter24_from_3d", "output")

# centerlines and outlet names
f_centerline = os.path.join(FDIR, "data", "centerlines_pfaller22")

# 0d-3d comparison error metrics
f_error_0d3d_geo = os.path.join(FDIR, "data", "0d_3d_comparison_geometric_pfaller22.json")
f_error_0d3d_cali = os.path.join(FDIR, "data", "0d_3d_comparison_calibrated_richter24.json")
