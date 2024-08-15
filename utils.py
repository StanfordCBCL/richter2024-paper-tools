#!/usr/bin/env python
# coding=utf-8

import os
import json
import numpy as np
from collections import defaultdict

FDIR = os.path.dirname(__file__)

# output path for all png's
f_out_png = os.path.join(FDIR, "png")
f_out_svg = os.path.join(FDIR, "svg")
f_out_pdf = os.path.join(FDIR, "pdf")

# list of geometries used in pfaller22 and richter24
f_geometries = os.path.join(FDIR, "geometries_vmr_pfaller22.txt")

# database for geometries
f_database = os.path.join(FDIR, "data", "vmr_models.json")

# model pictures
f_picture = os.path.join(FDIR, "data", "pictures")

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
f_e_0d3d_geo = os.path.join(FDIR, "data", "0d_3d_comparison_geometric_pfaller22.json")
f_e_0d3d_cali = os.path.join(FDIR, "data", "0d_3d_comparison_calibrated_richter24.json")

# set model colors
model_colors = {
    "Coronary": "brown",
    "Aortofemoral": "darkviolet",
    "Aorta": "darkgreen",
    "Animal and Misc": "crimson",
    "Pulmonary": "navy",
    "Congenital Heart Disease": "orange",
}

# highighted models
models_special = ["0104_0001", "0140_2001", "0080_0001"]

def get_geometries():
    # get geometries
    geos = np.loadtxt(f_geometries, dtype="str")

    # load model database
    with open(f_database, "r") as file:
        db = json.load(file)
    
    categories = defaultdict(list)
    for g in geos:
        categories[db[g]["params"]["deliverable_category"]] += [g]
    
    order = ["Animal and Misc", "Aorta", "Aortofemoral", "Coronary", "Congenital Heart Disease", "Pulmonary"]
    geos_sorted = []
    cats_sorted = []
    for o in order:
        geos_sorted += categories[o]
        cats_sorted += [o] * len(categories[o])
    return np.array(geos_sorted), np.array(cats_sorted)