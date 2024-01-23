#!/usr/bin/env python
# coding=utf-8

import pdb
import json
import os
import copy
import glob
import vtk
import argparse
import numpy as np
import svzerodplus
from scipy.interpolate import CubicSpline

from test_integration_cpp import run_test_case_by_name

this_file_dir = os.path.dirname(__file__)

# Number of observations to extract in result refinement
NUM_OBS = 100

def read_geo(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh

    Returns:
        vtk reader, point data, cell data
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader

def refine_with_cubic_spline_derivative(
    x: np.ndarray, y: np.ndarray, num: int
) -> np.ndarray:
    """Refine a curve using cubic spline interpolation with derivative.

    Args:
        x: X-coordinates
        y: Y-coordinates
        num: New number of points of the refined data.


    Returns:
        new_y: New y-coordinates
        new_dy: New dy-coordinates
    """
    y = y.copy()
    y[-1] = y[0]
    x_new = np.linspace(x[0], x[-1], num)
    spline = CubicSpline(x, y, bc_type="periodic")
    new_y = spline(x_new)
    new_dy = spline.derivative()(x_new)
    return new_y.tolist(), new_dy.tolist()


def collect_results_from_0d(inp, res):
    out = {"y": {}, "dy": {}}
    for f in ["flow", "pressure"]:
        # loop over all junctions
        for junction in inp["junctions"]:
            # junction inlets
            for br in junction["inlet_vessels"]:
                for vessel in inp["vessels"]:
                    if vessel["vessel_id"] == br:
                        name = ":".join(
                            [f, vessel["vessel_name"], junction["junction_name"]]
                        )
                        m = np.array(
                            res[f + "_out"][res["name"] == vessel["vessel_name"]]
                        )
                        time = np.array(
                            res["time"][res["name"] == vessel["vessel_name"]]
                        )
                        (
                            out["y"][name],
                            out["dy"][name],
                        ) = refine_with_cubic_spline_derivative(time, m, NUM_OBS)
            # junction outlets
            for br in junction["outlet_vessels"]:
                for vessel in inp["vessels"]:
                    if vessel["vessel_id"] == br:
                        name = ":".join(
                            [f, junction["junction_name"], vessel["vessel_name"]]
                        )
                        m = np.array(
                            res[f + "_in"][res["name"] == vessel["vessel_name"]]
                        )
                        time = np.array(
                            res["time"][res["name"] == vessel["vessel_name"]]
                        )
                        (
                            out["y"][name],
                            out["dy"][name],
                        ) = refine_with_cubic_spline_derivative(time, m, NUM_OBS)
        # loop over all boundary conditions
        for vessel in inp["vessels"]:
            if "boundary_conditions" in vessel:
                # inlets
                if "inlet" in vessel["boundary_conditions"]:
                    name = ":".join(
                        [
                            f,
                            vessel["boundary_conditions"]["inlet"],
                            vessel["vessel_name"],
                        ]
                    )
                    m = np.array(res[f + "_in"][res["name"] == vessel["vessel_name"]])
                    time = np.array(res["time"][res["name"] == vessel["vessel_name"]])
                    (
                        out["y"][name],
                        out["dy"][name],
                    ) = refine_with_cubic_spline_derivative(time, m, NUM_OBS)
                # outlets
                if "outlet" in vessel["boundary_conditions"]:
                    name = ":".join(
                        [
                            f,
                            vessel["vessel_name"],
                            vessel["boundary_conditions"]["outlet"],
                        ]
                    )
                    m = np.array(res[f + "_out"][res["name"] == vessel["vessel_name"]])
                    time = np.array(res["time"][res["name"] == vessel["vessel_name"]])
                    (
                        out["y"][name],
                        out["dy"][name],
                    ) = refine_with_cubic_spline_derivative(time, m, NUM_OBS)
    return out


if __name__ == "__main__":
    fname = "/Users/pfaller/Downloads/3d_rerun/0075_1001.vtp"

    geo = read_geo(fname).GetOutput()

    pdb.set_trace()