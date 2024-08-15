#!/usr/bin/env python

import pdb
import os
import json

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman"}
)

from utils import (
    f_out_png,
    f_out_svg,
    f_out_pdf,
    f_e_0d3d_geo,
    f_e_0d3d_cali,
    models_special,
    get_geometries,
)


def add_median_labels(ax: plt.Axes) -> None:
    """Add text labels to the median lines of a seaborn boxplot.
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    lines_per_box = len(lines) // len(boxes)
    i = 0
    for median in lines[4::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())

        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y

        # workaround for LaTeX
        f = "{:.1f}".format(value*100) + "\,\%"
        if i == 0:
            ha = "left"
            dx = 0.3
        else:
            ha = "right"
            dx = -0.4
        x += 0.11 + dx
        ax.text(
            x,
            y,
            f,
            ha=ha,
            va="center",
            color="black",
            fontsize=9,
        )
        i += 1


def print_error(sorting):
    # load post-processed error analysis
    err = {}
    with open(f_e_0d3d_geo, "r") as f:
        err["geometric"] = json.load(f)
    with open(f_e_0d3d_cali, "r") as f:
        err["optimized"] = json.load(f)

    # load model database
    geos, cats = get_geometries()
    if sorting == "alphabetical":
        order = np.argsort(geos)
        geos = geos[order]
        cats = cats[order]

    xtick = np.arange(len(geos))

    # select errors to plot
    fields = ["pressure", "flow"]
    domain = ["cap"]  # , "int"
    metric0 = ["max"]  # , "avg", "max" , "sys", "dia"
    metric1 = ["rel"]  # , "abs"

    # generate plots
    for d in domain:
        for m0 in metric0:
            for m1 in metric1:
                values = {}
                for f in fields:
                    values[f] = {}
                    for s in ["geometric", "optimized"]:
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
                    sharey=m1 == "rel",
                    width_ratios=[15, 1],
                )
                plot_bar_arrow(
                    fig1, ax1, xtick, values, geos, cats, m0, m1, f, d, f_out_png, f_out_svg, f_out_pdf, sorting
                )


def plot_bar_arrow(fig1, axes, xtick, values, labels, cats, m0, m1, f, d, folder_png, folder_svg, folder_pdf, name):
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
        "sys": "systolic",
        "dia": "diastolic",
    }
    categories = {
        "Animal and Misc": "Aorta",
        "Aorta": "Aorta",
        "Aortofemoral": "Aortofemoral",
        "Congenital Heart Disease": "Pulmonary",
        "Coronary": "Coronary",
        "Pulmonary": "Pulmonary",
    }

    # combine some categories
    cats = np.array([categories[c] for c in cats])

    # create gap between categories
    unique_cats = np.unique(cats)
    offset = [0]
    skip = 0.5
    for i in range(1, len(cats)):
        j0 = cats.tolist().index(cats[i - 1])
        j1 = cats.tolist().index(cats[i])
        offset += [offset[i - 1] + int(j0 != j1) * skip]
    positions = xtick + np.array(offset)
    xlim = [-skip, len(labels) + (unique_cats.size - 1) * skip - skip]

    # collect error change
    diff_raw = []
    for f in fields:
        diff_raw += [
            np.log(values[f]["geometric"] / values[f]["optimized"]) / np.log(10)
        ]
    diff = np.array(diff_raw)

    # set colors according to error change
    cmap_p = "Blues"
    cmap_m = "Reds"
    diff_p = diff >= 0
    diff_m = diff < 0
    diff[diff_p] = 0.5 + (diff[diff_p]) / diff.max() / 1.5
    diff[diff_m] = 0.5 + (diff[diff_m]) / diff.min() / 2.5
    colors = np.zeros(diff.shape + (4,))
    colors[diff_p] = plt.colormaps[cmap_p](diff[diff_p])
    colors[diff_m] = plt.colormaps[cmap_m](diff[diff_m])

    for j, (f, ax) in enumerate(zip(fields, axes)):
        # general settings
        for i in range(2):
            ax[i].cla()
            ax[i].spines["top"].set_visible(True)
            ax[i].spines["right"].set_visible(True)
            ax[i].yaxis.grid("both")
            ax[i].yaxis.set_ticks_position("both")
            if j == 1:
                labels = np.array(labels, dtype=object)
                for k in range(len(labels)):
                    if labels[k] in models_special:
                        labels[k] = "$\\Rightarrow$ " + labels[k]
                ax[i].set_xticks(positions, labels, rotation="vertical", fontsize=12, va="top")
            if m1 == "rel":
                ax[i].set_yscale("log")
                ax[i].yaxis.set_major_formatter(
                    mtick.PercentFormatter(xmax=1, decimals=1)
                )
                ax[i].tick_params(axis="y", labelsize=12)
            # else:
            #     values[f]["geometric"] *= convert[f]
            #     values[f]["optimized"] *= convert[f]

        # plot geometries
        data = zip(values[f]["geometric"], values[f]["optimized"], colors[j])
        for i, (val0, val1, col) in enumerate(data):
            pos = positions[i]
            mar = "v" if val0 > val1 else "^"
            ax[0].plot([pos], [val1], color=col, marker=mar, markersize=6)
            ax[0].plot([pos, pos], [val0, val1], color=col, linewidth=2)

        ax[0].xaxis.grid("both")
        ax[0].set_xlim(xlim)
        title = "Change in " + names[m0] + " " + f + " error"
        if m1 == "abs":
            title += " [" + units[f] + "]"
        ax[0].set_title(title, fontsize=15)
        if j == 0:
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
                    xytext=(0, -60),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                )

        # plot box plots
        df_geo = pd.DataFrame({f: values[f]["geometric"], "Model": "Geometric"})
        df_opt = pd.DataFrame({f: values[f]["optimized"], "Model": "Optimized"})
        df_box = pd.concat([df_geo, df_opt])
        sns.boxplot(
            x="Model",
            y=f,
            data=df_box,
            hue="Model",
            legend=False,
            ax=ax[1],
            palette="YlGn",
            linewidth=1,
            width=0.5,
            fliersize=0,
            saturation=1,
        )
        add_median_labels(ax[1])
        ax[1].set_xlabel("")
        ax[1].tick_params(axis="y", labelsize=12)
        ax[1].yaxis.tick_right()
        ax[1].set_xlim([-0.5, 1.5])
        ax[1].set_xticks(
            [0, 1], ["Geometric", "Calibrated"], rotation="vertical", fontsize=12
        )

    plt.subplots_adjust(hspace=0.05)

    fname1 = os.path.join(
        folder_png, "error_arrow_" + name + "_" + d + "_" + m0 + "_" + m1 + ".png"
    )
    fname2 = os.path.join(
        folder_svg, "error_arrow_" + name + "_" + d + "_" + m0 + "_" + m1 + ".svg"
    )
    fname3 = os.path.join(
        folder_pdf, "error_arrow_" + name + "_" + d + "_" + m0 + "_" + m1 + ".pdf"
    )
    plt.tight_layout()  # rect=(0, 0, 0.8, 1))
    for fname in [fname1, fname2, fname3]:
        fig1.savefig(fname, bbox_inches="tight")
        print(fname)
    plt.close()


if __name__ == "__main__":
    for sorting in ["categories"]:  # , "alphabetical"
        print_error(sorting)
