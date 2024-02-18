#!/usr/bin/env python

import os
import json
import sys
import pdb

import numpy as np
import pandas as pd

from collections import defaultdict
from scipy.interpolate import interp1d

# svZeroDSolver from https://github.com/SimVascular/svZeroDSolver
import pysvzerod

from utils import f_geometries, f_geo_in, f_geo_out, f_centerline
from simulation_io import get_caps


def run(inp, out, overwrite=False):
    # run the simulation (save results to disk)
    sys.argv = ["run_simulation_cli", inp, out]
    if(not os.path.exists(out) or (os.path.exists(out) and overwrite)):
        pysvzerod.run_simulation_cli()

    # read the results from disk
    return pd.read_csv(out)


def calc_error(geo, res, time=None):
    # simulations to compare
    m_rom = "0d"
    m_ref = "3d_rerun"

    # set post-processing fields
    fields = ['flow', 'pressure']

    # get 1d/3d map
    f_cent = os.path.join(f_centerline, geo + ".vtp")
    f_outlet = os.path.join(f_centerline, "outlets_" + geo)
    caps = get_caps(f_outlet, f_cent)
    pdb.set_trace()

    # interpolate 1d to 3d in space and time (allow extrapolation due to round-off errors at bounds)
    interp = lambda x_1d, y_1d, x_3d: interp1d(x_1d, y_1d.T, fill_value='extrapolate')(x_3d)

    # branches (interior and caps)
    branches = {'int': {}, 'cap': {}}
    for f in fields:
        branches['int'][f] = list(res.keys())
        branches['cap'][f] = list(caps.values())

    # skip inlet branch for flow (flow is prescribed)
    for d in branches.keys():
        if len(branches[d]['flow']) > 1:
            branches[d]['flow'].remove(0)

    # pick systole and diastole time points
    inflow = res[0]['flow'][m_ref + '_cap_last']
    times = {'sys': np.argmax(inflow), 'dia': np.argmin(inflow)}

    # get spatial errors
    err = rec_dict()
    for f in fields:
        # calculate maximum delta over whole model over all time steps
        res_all = res[0][f][m_ref + '_int_last']
        for br in branches['int'][f][1:]:
            res_all = np.vstack((res_all, res[br][f][m_ref + '_int_last']))
        norm_delta = np.max((np.max(res_all, axis=0) - np.min(res_all, axis=0)))

        for br in branches['int'][f]:
            # retrieve 3d results
            res_3d = res[br][f][m_ref + '_int_last']

            # map paths to interval [0, 1]
            path_1d = res[br][m_rom + '_path'] / res[br][m_rom + '_path'][-1]
            path_3d = res[br][m_ref + '_path'] / res[br][m_ref + '_path'][-1]

            # interpolate in space and time
            res_1d = interp(path_1d, res[br][f][m_rom + '_int_last'], path_3d)
            res_1d = interp(time[m_rom], res_1d, time[m_ref])

            # calculate spatial error (eliminate time dimension)
            if f == 'flow' or (f == 'area' and m_rom == '1d'):
                norm = np.max(res_3d, axis=1) - np.min(res_3d, axis=1)
            else:
                norm = np.mean(res_3d, axis=1)

            # difference between ref and rom
            diff = np.abs(res_1d - res_3d).T

            # loop through different error metrics
            for n in ['abs', 'rel', 'rel_delta']:
                if n == 'rel':
                    delta = diff / norm
                elif n == 'rel_delta':
                    delta = diff / norm_delta
                else:
                    delta = diff
                for tn, ti in times.items():
                    err[f]['spatial'][tn][n][br] = delta[ti]
                err[f]['spatial']['avg'][n][br] = np.mean(delta, axis=0)
                err[f]['spatial']['max'][n][br] = np.max(delta, axis=0)

    # get mean errors
    for f in fields:
        for m in err[f]['spatial'].keys():
            for n in err[f]['spatial'][m].keys():
                # get interior error
                for br in branches['int'][f]:
                    err[f]['int'][m][n][br] = np.mean(err[f]['spatial'][m][n][br])

                # get cap error
                for br in branches['cap'][f]:
                    if len(branches['cap'][f]) > 1 and br == 0:
                        # inlet
                        i_cap = 0
                    else:
                        # outlet
                        i_cap = -1
                    err[f]['cap'][m][n][br] = err[f]['spatial'][m][n][br][i_cap]

                # get error over all branches
                err[f]['int'][m][n]['all'] = np.mean([err[f]['int'][m][n][br] for br in branches['int'][f]])
                err[f]['cap'][m][n]['all'] = np.mean([err[f]['cap'][m][n][br] for br in branches['cap'][f]])
    return err


def rec_dict():
    """
    Recursive defaultdict
    """
    return defaultdict(rec_dict)


def compare(fname):
    print(fname)

    # set input and output paths
    inp = os.path.join(f_geo_in, fname + ".json")
    out = os.path.join(f_geo_out, fname + ".csv")
    
    # run 1d simulation
    res = run(inp, out)

    # calculate 0d-3d approximation error metrics
    err = calc_error(fname, res)

    pdb.set_trace()


if __name__ == "__main__":
    # loop over all vmr models
    files = np.loadtxt(f_geometries, dtype="str")
    for f in files:
        compare(f)
