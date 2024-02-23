#!/usr/bin/env python
# coding=utf-8

import pdb
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    f_out,
    f_geo_in,
    f_cali_0d_out,
    f_cali_3d_out,
    models_special,
    get_geometries,
)

# use LaTeX in text
plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman"}
)


def plot(dim):
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

    # set element properties
    elements = ["R_poiseuille", "stenosis_coefficient", "L", "C"]
    colors = {}
    for i, e in enumerate(elements):
        colors[e[0]] = plt.cm.Dark2(i)
    elements_pos = [[-1, 0], [0, 0], [-1, -1], [0, -1]]

    fig, ax = plt.subplots(nx, ny, figsize=(ny * 2, nx * 2.5), dpi=300)
    for j, (fname, cat) in enumerate(zip(files, cats)):
        ab = np.unravel_index(j, (nx, ny))

        # read results
        with open(os.path.join(f_geo_in, fname + ".json")) as f:
            inp = json.load(f)
        with open(os.path.join(f_cali_out, fname + ".json")) as f:
            opt = json.load(f)

        # loop all 0D elements
        for j, ele in enumerate(elements):
            # collect elements from all vessels
            ref = []
            sol = []
            for i in range(len(inp["vessels"])):
                rval = inp["vessels"][i]["zero_d_element_values"][ele]
                sval = opt["vessels"][i]["zero_d_element_values"][ele]
                ref += [rval]
                sol += [sval]
            ref = np.array(ref)
            sol = np.array(sol)

            # manually set limits
            xlim = [ref.min(), ref.max()]
            ylim = [sol.min(), sol.max()]
            lim = np.array([np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])])

            # add margin so all dots are fully within plot
            eps = 0.1
            delta = np.diff(lim) * eps
            lim[0] -= delta
            lim[1] += delta

            # normalize parameter range
            ref = (ref - lim.min()) / (lim.max() - lim.min())
            sol = (sol - lim.min()) / (lim.max() - lim.min())

            # put plot in the correct quadrant
            offset = elements_pos[j]

            ax[ab].scatter(ref + offset[0], sol + offset[1], s=50, color=colors[ele[0]])
            ax[ab].set_xticklabels([])
            ax[ab].set_yticklabels([])
            ax[ab].spines["top"].set_visible(True)
            ax[ab].spines["right"].set_visible(True)
            ax[ab].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # plot diagonal
            diag = np.array([0, 1])
            ax[ab].plot(diag + offset[0], diag + offset[1], "k--", linewidth=1.5)

        # plot dividers
        ax[ab].plot([-1, 1], [0, 0], "k-")
        ax[ab].plot([0, 0], [-1, 1], "k-")

        ax[ab].set_aspect("equal", adjustable="box")

        total_lim = [-1, 1]
        ax[ab].set_xlim(total_lim)
        ax[ab].set_ylim(total_lim)
        if fname in models_special:
            col = "r"
            title = "$\\textbf{" + fname + "}$"
        else:
            col = "k"
            title = fname
        ax[ab].set_title(title, fontsize=21, color=col)

        plt.subplots_adjust(
            left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2
        )

    xtext = "Geometric 0D elements"
    ytext = "Optimized 0D elements from " + str(dim) + "D results"
    fig.text(0.5, -0.01, xtext, ha="center", fontsize=24)
    fig.text(-0.01, 0.5, ytext, va="center", fontsize=24, rotation="vertical")
    plt.tight_layout()
    fout = os.path.join(f_out, "optimized_" + str(dim) + "d.png")
    fig.savefig(fout, bbox_inches="tight")
    print(fout)
    plt.close(fig)


if __name__ == "__main__":
    # plot 0d element correlation: geometric vs. calibrated from 3d
    plot(3)

    # plot 0d element correlation: geometric vs. calibrated from 0d
    plot(0)
