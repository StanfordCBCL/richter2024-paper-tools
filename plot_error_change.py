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
    metric0 = ["max", "avg"]  # , 'sys', 'dia'
    metric1 = ["rel", "abs"]

    # generate plots
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

                fig1, ax1 = plt.subplots(
                    2,
                    2,
                    dpi=300,
                    figsize=(12, 6),
                    sharex="col",
                    width_ratios=[20, 1],
                )
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
    categories = {
        "Animal and Misc": "Animal",
        "Aorta": "Aorta",
        "Aortofemoral": "Aortofemoral",
        "Congenital Heart Disease": "CHD",
        "Coronary": "Coronary",
        "Pulmonary": "Pulmonary",
    }
    models = ["0104_0001", "0140_2001", "0080_0001"]

    # create gap between categories
    unique_cats = np.unique(cats)
    xlim = [-1, len(labels) + unique_cats.size - 1]
    offset = [0]
    for i in range(1, len(cats)):
        j0 = cats.tolist().index(cats[i - 1])
        j1 = cats.tolist().index(cats[i])
        offset += [offset[i - 1] + int(j0 != j1)]
    positions = xtick + np.array(offset)

    for j, (f, ax) in enumerate(zip(fields, axes)):
        # general settings
        for i in range(2):
            ax[i].cla()
            ax[i].spines["top"].set_visible(True)
            ax[i].spines["right"].set_visible(True)
            ax[i].yaxis.grid("both")
            ax[i].yaxis.set_ticks_position("both")
            if j == 1:
                ax[i].set_xticks(positions, labels, rotation="vertical")
                for label in ax[0].get_xticklabels():
                    if label.get_text() in models:
                        label.set_color("r")
            if m1 == "rel":
                ax[i].set_yscale("log")
                ax[i].yaxis.set_major_formatter(
                    mtick.PercentFormatter(xmax=1, decimals=1)
                )
            # else:
            #     values[f]["geometric"] *= convert[f]
            #     values[f]["calibrated"] *= convert[f]

        # plot geometries
        data = zip(values[f]["geometric"], values[f]["calibrated"], cats)
        for i, (val0, val1, cat) in enumerate(data):
            pos = positions[i]
            if val0 > val1:
                m = "v"
            else:
                m = "^"
            if labels[i] in models:
                col = "r"
                width = 1.5
            else:
                col = "k"
                width = 1
            ax[0].plot([pos], [val1], color=col, marker=m, markersize=3)
            ax[0].plot([pos, pos], [val0, val1], color=col, linewidth=width)

        ax[0].xaxis.grid("both")
        ax[0].set_xlim(xlim)
        ylabel = f.capitalize()
        if m1 == "abs":
            ylabel += " [" + units[f] + "]"
        ax[0].set_ylabel(ylabel)
        if j == 0:
            ax[0].set_title(
                names[m0].capitalize() + " " + names[m1] + " error at " + names[d]
            )
            ax[0].tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )

        # add model categories
        if name == "categories" and j == 1:
            for cat in unique_cats:
                mid_point = np.mean(positions[cats == cat])
                ax[0].annotate(
                    categories[cat],
                    xy=(mid_point, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -45),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                )

        # plot box plots
        ax[1].boxplot(
            values[f]["geometric"], positions=[0], widths=0.6, labels=["Geometric"]
        )
        ax[1].boxplot(
            values[f]["calibrated"], positions=[1], widths=0.6, labels=["Calibrated"]
        )
        ax[1].yaxis.tick_right()

    plt.subplots_adjust(hspace=0.05)
    fname = os.path.join(
        folder, "error_arrow_" + name + "_" + d + "_" + m0 + "_" + m1 + ".png"
    )
    print(fname)
    plt.tight_layout(rect=(0, 0, 0.8, 1))
    fig1.savefig(fname, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for sorting in ["categories", "alphabetical"]:
        print_error(sorting)
