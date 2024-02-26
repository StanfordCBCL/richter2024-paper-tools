#!/usr/bin/env python
# coding=utf-8

import pdb
import scipy
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import (
    f_out,
    f_picture,
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


def add_img(ax, geo):
    # load image
    impath = os.path.join(f_picture, geo + '.png')
    img = plt.imread(impath)

    # Determine whether to scale based on width or height
    aspect_ratio = img.shape[1] / img.shape[0]
    if aspect_ratio > 1:
        target_width = 1
        target_height = target_width / aspect_ratio
    else:
        target_height = 1
        target_width = target_height * aspect_ratio

    # Calculate the center of the target box
    center_x = (-3 + -2) / 2
    center_y = (0 + 1) / 2

    # Adjust the extent based on the calculated center and target size
    left = center_x - target_width / 2
    right = center_x + target_width / 2
    bottom = center_y - target_height / 2
    top = center_y + target_height / 2

    ax.imshow(img, extent=[left, right, bottom, top], aspect='auto')


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
    nv = 18
    nh = 4

    # set element properties
    elements = ["R_poiseuille", "L", "C", "stenosis_coefficient"]
    colors = {}
    for i, e in enumerate(elements):
        colors[e[0]] = plt.cm.Dark2(i)
    elements_pos = np.array([[-2, 0], [-1, 0], [0, 0], [1, 0]])

    fig, ax = plt.subplots(nv, nh, figsize=(nh * 3, nv), dpi=500)

    correlations = defaultdict(list)
    for j, (fname, cat) in enumerate(zip(files, cats)):
        ab = np.unravel_index(j, (nv, nh))

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

            # get some statistics
            _, _, r_value, _, _ = scipy.stats.linregress(ref, sol)
            correlations[ele] += [r_value**2]

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

            # plot diagonal
            diag = np.array([0, 1])
            ax[ab].plot(diag + offset[0], diag + offset[1], "k--", linewidth=1.5)

        # add model image
        add_img(ax[ab], fname)

        # plot dividers
        for pos in elements_pos:
            px = [pos[0], pos[0]]
            py = [pos[1], pos[1] + 1.0]
            ax[ab].plot(px, py, "k-", linewidth=1.5)

        ax[ab].set_aspect("equal", adjustable="box")
        ax[ab].set_xticklabels([])
        ax[ab].set_yticklabels([])
        ax[ab].spines["top"].set_visible(True)
        ax[ab].spines["right"].set_visible(True)
        ax[ab].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[ab].set_xlim([-3, 2])
        ax[ab].set_ylim([0, 1])
        col = "k"
        title = fname
        # if fname in models_special:
        #     col = "r"
        #     title = "$\\textbf{" + fname + "}$"
        ax[ab].set_title(title, fontsize=21, color=col)

    # print correlations
    print(str(dim) + "D r_value (mean, std)")
    for k, v in correlations.items():
        print(k, np.mean(v), np.std(v))

    xtext = "Geometric 0D elements"
    ytext = "Optimized 0D elements from " + str(dim) + "D results"
    fig.text(0.5, -0.01, xtext, ha="center", fontsize=24)
    fig.text(-0.02, 0.5, ytext, va="center", fontsize=24, rotation="vertical")
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
