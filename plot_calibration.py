#!/usr/bin/env python
# coding=utf-8

import pdb
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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
    elements = ["R_poiseuille", "stenosis_coefficient", "L", "C"]

    fig = plt.figure(figsize=(ny * 2, nx * 2), dpi=300)
    main_gs = GridSpec(nx, ny, figure=fig)
    for j, (fname, cat) in enumerate(zip(files, cats)):
        print(fname)
        row, col = divmod(j, ny)

        # read results
        with open(os.path.join(f_geo_in, fname + ".json")) as f:
            inp = json.load(f)
        with open(os.path.join(f_cali_out, fname + ".json")) as f:
            opt = json.load(f)

         # Create a 2x2 grid for each subplot
        sub_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=main_gs[row, col], wspace=0, hspace=0)

        for j, ele in enumerate(elements):
            ax = fig.add_subplot(sub_gs[j])
            ref = []
            sol = []
            col = []
            for i in range(len(inp["vessels"])):
                rval = inp["vessels"][i]["zero_d_element_values"][ele]
                sval = opt["vessels"][i]["zero_d_element_values"][ele]
                ref += [rval]
                sol += [sval]
                col += [colors[ele[0]]]
            if j == 0:
                ax.set_title(fname, fontweight="bold", fontsize=12)
            ax.scatter(ref, sol, s=20, c=col)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # manually set limits
            xlim = [np.min(ref), np.max(ref)]
            ylim = [np.min(sol), np.max(sol)]
            lim = [np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])]

            # add margin so all dots are fully within plot
            eps = 0.1
            delta = np.diff(lim) * eps
            lim[0] -= delta
            lim[1] += delta
            ax.set_xlim(lim)
            ax.set_ylim(lim)

            # plot diagonal
            ax.plot(lim, lim, "k--")

            ax.set_aspect("equal", adjustable="box")
            plt.subplots_adjust(wspace=0, hspace=0)

    xtext = "Geometric 0D elements"
    ytext = "Calibrated 0D elements from " + str(dim) + "D results"
    fig.text(0.5, -0.01, xtext, ha="center", fontsize=24)
    fig.text(-0.01, 0.5, ytext, va="center", fontsize=24, rotation="vertical")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    fout = os.path.join(f_out, "calibration_" + str(dim) + "d.png")
    fig.savefig(fout, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # plot 0d element correlation: geometric vs. calibrated from 3d
    plot(3)

    # plot 0d element correlation: geometric vs. calibrated from 0d
    plot(0)
