#!/usr/bin/env python
# coding=utf-8

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import f_out, f_geo_in, f_cali_0d_out, f_cali_3d_out, model_colors, get_geometries

# use LaTeX in text
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman"}
)


def plot(dim):
    print("\nComparing 0D-" + str(dim) + "D\n")
    if dim == 0:
        f_cali_out = f_cali_0d_out
    elif dim == 3:
        f_cali_out = f_cali_3d_out
    else:
        raise ValueError("Unknown dimension " + str(dim))

    # get geometries and colors
    files, cats = get_geometries()

    # compare 0D element values
    nx = 9
    ny = 8
    colors = {"C": "b", "L": "g", "R": "r", "s": "k"}
    elements = ["R_poiseuille", "C", "L", "stenosis_coefficient"]

    fig, ax = plt.subplots(nx, ny, figsize=(ny * 2, nx * 2), dpi=300)
    for j, (fname, cat) in enumerate(zip(files, cats)):
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
        ax[ab].set_title(fname, color=model_colors[cat], fontweight="bold", fontsize=12)
        # ax[ab].grid(True)
        ax[ab].scatter(ref, sol, s=20, c=col)
        ax[ab].set_yscale("log")
        ax[ab].set_xscale("log")
        ax[ab].set_xticklabels([])
        ax[ab].set_yticklabels([])
        ax[ab].spines["top"].set_visible(True)
        ax[ab].spines["right"].set_visible(True)

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
    fout = os.path.join(f_out, "calibration_" + str(dim) + "d.png")
    fig.savefig(fout, bbox_inches="tight")


if __name__ == "__main__":
    # plot 0d element correlation: geometric vs. calibrated from 0d
    plot(0)

    # plot 0d element correlation: geometric vs. calibrated from 3d
    plot(3)
