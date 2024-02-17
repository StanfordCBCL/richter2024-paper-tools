#!/usr/bin/env python

import pdb
import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.path as mpath
from matplotlib.patches import FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman"}
)

from utils import (
    f_out,
    f_e_0d3d_geo,
    f_e_0d3d_cali,
    model_colors,
    get_geometries,
)


def print_error(sorting):
    # load post-processed error analysis
    with open(f_e_0d3d_geo, "r") as f:
        geometric = json.load(f)
    with open(f_e_0d3d_cali, "r") as f:
        calibrated = json.load(f)
    err = {"geometric": geometric, "calibrated": calibrated}

    # load model database
    geos, cats = get_geometries()
    if sorting == "alphabetical":
        order = np.argsort(geos)
        geos = geos[order]
        cats = cats[order]

    xtick = np.arange(len(geos))

    # select errors to plot
    fields = ["pressure", "flow"]
    domain = ["cap", "int"]
    metric0 = ["avg", "max"]  # , 'sys', 'dia'
    metric1 = ["rel", "abs"]

    # generate plots
    fig1, ax1 = plt.subplots(2, 1, dpi=300, figsize=(12, 6), sharex=True)
    for d in domain:
        for m0 in metric0:
            for m1 in metric1:
                values = {}
                for f in fields:
                    values[f] = {}
                    for s in ["geometric", "calibrated"]:
                        values[f][s] = []
                        for k in geos:
                            values[f][s] += [err[s][k][f][d][m0][m1]["all"]]
                        values[f][s] = np.array(values[f][s])

                plot_bar_arrow(
                    fig1, ax1, xtick, values, geos, cats, m0, m1, f, d, f_out, sorting
                )


def plot_bar_arrow(fig1, axes, xtick, values, labels, cats, m0, m1, f, d, folder, name):
    # unit conversion
    fields = ["pressure", "flow"]
    units = {"pressure": "mmHg", "flow": "l/min", "area": "mm$^2$"}
    cgs2mmhg = 7.50062e-4
    mlps2lpmin = 60.0 / 1000.0
    convert = {"pressure": cgs2mmhg, "flow": mlps2lpmin, "area": 100}
    names = {
        "avg": "average",
        "max": "maximum",
        "rel": "relative",
        "abs": "absolute",
        "cap": "caps",
        "int": "interior",
    }

    plt.cla()
    xlim = [-1, len(labels)]

    for j, (f, ax) in enumerate(zip(fields, axes)):
        ax.cla()
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

        if m1 == "rel":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

        # plot average
        means = {}
        for k in values[f].keys():
            if m1 == "abs":
                values[f][k] *= convert[f]
            means[k] = np.mean(values[f][k])
            ax.plot(xlim, [means[k]] * 2, color="0.5")

        # plot geometries
        data = zip(values[f]["geometric"], values[f]["calibrated"], cats)
        for i, (val0, val1, cat) in enumerate(data):
            if val0 > val1:
                m = "v"
            else:
                m = "^"
            ax.plot([i], [val1], color=model_colors[cat], marker=m, markersize=4)
            ax.plot([i, i], [val0, val1], color=model_colors[cat], linewidth=2)

        ax.set_xlim(xlim)
        ax.xaxis.grid("both")
        ax.yaxis.grid("both")
        ylabel = f.capitalize()
        if m1 == "abs":
            ylabel += " [" + units[f] + "]"
        ax.set_ylabel(ylabel)
        if j == 0:
            ax.set_title(
                names[m0].capitalize() + " " + names[m1] + " error at " + names[d]
            )
            ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
        if j == 1:
            ax.set_xticks(xtick, labels, rotation="vertical")

    plt.subplots_adjust(hspace=0.05)
    fname = os.path.join(
        folder, "error_arrow_" + name + "_" + d + "_" + m0 + "_" + m1 + ".png"
    )
    ratio = means["geometric"] / means["calibrated"]
    print(fname)
    for f in fields:
        ratio = values[f]["geometric"] / values[f]["calibrated"]
        print("error reduction ", f, np.mean(ratio))
    plt.tight_layout(rect=(0, 0, 0.8, 1))
    fig1.savefig(fname, bbox_inches="tight")


if __name__ == "__main__":
    for sorting in ["alphabetical", "categories"]:
        print_error(sorting)
