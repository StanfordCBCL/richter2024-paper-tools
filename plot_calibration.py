#!/usr/bin/env python
# coding=utf-8

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import f_geometries, f_geo_in, f_cali_0d_out, f_cali_3d_out

# use LaTeX in text
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.size": 12,
    }
)


def plot(dim):
    print("\nComparing 0D-" + str(dim) + "D\n")
    if dim == 0:
        f_cali_out = f_cali_0d_out
    elif dim == 3:
        f_cali_out = f_cali_3d_out
    else:
        raise ValueError("Unknown dimension " + str(dim))

    files = np.loadtxt(f_geometries, dtype="str")
    assert len(files) == 72, "wrong number of files"

    # compare 0D element values
    nx = 9
    ny = 8
    colors = {"C": "b", "L": "g", "R": "r", "s": "k"}
    elements = ["R_poiseuille", "C", "L", "stenosis_coefficient"]

    fig, ax = plt.subplots(nx, ny, figsize=(ny * 2, nx * 2), dpi=300)
    for j, fname in enumerate(files):
        print(fname)
        ab = np.unravel_index(j, (nx, ny))

        # read results
        with open(os.path.join(f_geo_in, fname + ".json")) as f:
            inp = json.load(f)
        with open(os.path.join(f_cali_out, fname + ".json")) as f:
            opt = json.load(f)

        ref = []
        sol = []
        col = []
        for ele, cl in zip(elements, colors):
            for i in range(len(inp["vessels"])):
                if ele in inp["vessels"][i]["zero_d_element_values"]:
                    rval = inp["vessels"][i]["zero_d_element_values"][ele]
                    sval = opt["vessels"][i]["zero_d_element_values"][ele]
                    if rval > 0.0 and sval > 0.0:
                        ref += [rval]
                        sol += [sval]
                        col += [colors[ele[0]]]
                else:
                    continue
        if not ref:
            print("no values found")
            continue
        ax[ab].set_title(fname)
        # ax[ab].grid(True)
        ax[ab].scatter(ref, sol, s=20, c=col)
        ax[ab].set_yscale("log")
        ax[ab].set_xscale("log")
        ax[ab].set_xticklabels([])
        ax[ab].set_yticklabels([])

        # manually set limits
        eps = 10
        xlim = [np.min(ref) / eps, np.max(ref) * eps]
        ylim = [np.min(sol) / eps, np.max(sol) * eps]
        lim = [np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])]
        ax[ab].set_xlim(lim)
        ax[ab].set_ylim(lim)

        # plot diagonal
        ax[ab].plot(lim, lim, "k--")

        ax[ab].set_aspect("equal", adjustable="box")
    xtext = "Geometric 0D elements"
    ytext = "Calibrated 0D elements from " + str(dim) + "D results"
    fig.text(0.5, -0.01, xtext, ha="center", fontsize=24)
    fig.text(-0.01, 0.5, ytext, va="center", fontsize=24, rotation="vertical")
    plt.tight_layout()
    fout = os.path.join("png", "calibration_" + str(dim) + "d.png")
    fig.savefig(fout, bbox_inches="tight")


if __name__ == "__main__":
    # plot 0d element correlation: geometric vs. calibrated from 0d
    plot(0)

    # plot 0d element correlation: geometric vs. calibrated from 3d
    plot(3)
