#!/usr/bin/env python

import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman"}
)


def print_error():
    # load post-processed error analysis
    with open("0d_3d_comparison_geometric_pfaller22.json", "r") as f:
        geometric = json.load(f)
    with open("0d_3d_comparison_calibrated_richter24.json", "r") as f:
        calibrated = json.load(f)
    err = {"geometric": geometric, "calibrated": calibrated}
    geos = sorted(geometric.keys())

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
                values = []
                for f in fields:
                    values_s = []
                    for s, res in err.items():
                        values_f = []
                        for k in geos:
                            if k in res:
                                values_f += [res[k][f][d][m0][m1]["all"]]
                            else:
                                values_f += [np.nan]
                        values_s += [values_f]
                    values += [values_s]

                values = np.array(values)
                xtick = np.arange(len(values[0, 0]))

                plot_bar_arrow(
                    fig1, ax1, xtick, values, geos, m0, m1, f, d, "../png", "aplhabetical"
                )


def plot_bar_arrow(fig1, axes, xtick, values, labels, m0, m1, f, d, folder, name):
    m_rom = "0D"
    # unit conversion
    fields = ["pressure", "flow"]
    units = {"pressure": "mmHg", "flow": "l/min", "area": "mm$^2$"}
    cgs2mmhg = 7.50062e-4
    mlps2lpmin = 60.0 / 1000.0
    convert = {"pressure": cgs2mmhg, "flow": mlps2lpmin, "area": 100}

    plt.cla()
    xlim = [-1, len(labels)]

    for j, ax in enumerate(axes):
        ax.cla()

        if m1 == "rel":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
            values_plot = values
        elif m1 == "abs":
            values_plot = values * convert[f]

        col = "0.5"
        avg = np.mean(values_plot[j], axis=1)
        ax.plot(xlim, [avg[0]] * 2, color=col)
        ax.plot(xlim, [avg[1]] * 2, color=col)

        for i, (geo, val) in enumerate(zip(labels, values_plot[j].T)):
            if val[0] > val[1]:
                col = "g"
                m = r"$\downarrow$"
            else:
                col = "r"
                m = r"$\uparrow$"
            ax.plot([i], [val[1]], color=col, marker=m, markersize=8)
            ax.plot([i, i], [val[0], val[1]], color="k")

        ax.set_xlim(xlim)
        ax.xaxis.grid("both")
        ax.yaxis.grid("both")
        ax.set_ylabel(fields[j].capitalize())
        if j == 0:
            ax.set_title(m0 + " " + m1 + " error at " + d)
            ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
        if j == 1:
            ax.set_xticks(xtick, labels, rotation="vertical")

    plt.subplots_adjust(hspace=0.05)
    fname = os.path.join(
        folder, "error_arrow_" + name + "_" + d + "_" + m0 + "_" + m1 + ".png"
    )
    print("error reduction", fname, avg[0] / avg[1])
    fig1.savefig(fname, bbox_inches="tight")


if __name__ == "__main__":
    print_error()
