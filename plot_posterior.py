
import json
import os
import pickle

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from matplotlib.ticker import MaxNLocator
from rich import print
from svsuperestimator.reader import (CenterlineHandler, SimVascularProject,
                                     SvZeroDSolverInputHandler)
from svsuperestimator.tasks import taskutils
from svzerodsolver import runnercpp

# Make the plot



this_file_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_file_dir, "config.json")) as ff:
    global_config = json.load(ff)

target_folder = os.path.join(this_file_dir, "build", "anatomies")

os.makedirs(target_folder, exist_ok=True)

width = global_config["figure.figsize"][0] * 2
height = global_config["figure.figsize"][1] * 2
 
global_config["figure.figsize"] = (width, width*2/3)

matplotlib.rcParams.update(global_config)

cl_source = "/Volumes/richter/0107_10_variations"

# model_image_loc = {
#     "0003_0001": [0.56,-0.15,0.4,0.3],
#     "0097_0001": [0.5,-0.17,0.3,0.3],
#     "0107_0001": [0.6,0.-0.1,0.4,0.6],
# }


for model_name in ["0104_0001"]:


    source = f"/Volumes/richter/projects/{model_name}/ParameterEstimation"

    project = SimVascularProject(f"/Volumes/richter/projects/{model_name}")
    zerod_handler = project["0d_simulation_input"]


    metrics_all = {}

    folder = f"/Volumes/richter/projects/{model_name}/ParameterEstimation/multi_fidelity_lm_calibration2"
    # model_img = f"six_anatomies/bc_descs/{model_name}.png"

    windkessel_tunings = sorted([f for f in os.listdir(folder) if f.endswith("windkessel_tuning")], key=lambda x: int(x.split("_")[0]))

    # for wk_tuning in windkessel_tunings:

    with open(os.path.join(folder, windkessel_tunings[-1], "results.json")) as ff:
        data = json.load(ff)
    
    with open(os.path.join(folder, windkessel_tunings[-1], "taskdata.json")) as ff:
        taskdata = json.load(ff)

    gt = np.array(data["metrics"]["ground_truth"])
    map = np.array(data["metrics"]["maximum_a_posteriori"])

    particles = np.array(taskdata["particles"][-1])
    weights = np.array(taskdata["weights"][-1]).flatten()

    mean = np.average(particles, weights=weights, axis=0)
    covmat = np.cov(particles.T, aweights=weights)
    std = np.array([covmat[i][i] ** 0.5 for i in range(covmat.shape[0])])

    outlet_bcs = list(zerod_handler.outlet_boundary_conditions)

    labels = [r"$\theta^{" + f"({i+1})" + "}$" for i in range(len(outlet_bcs))]

    max_row = int(len(outlet_bcs) / 3) if len(outlet_bcs) % 3 == 0 else int(len(outlet_bcs) / 3) + 1

    fig, axs = plt.subplots(max_row, 3, figsize=(width, width*max_row/3 *0.55))

    for i in range(3-len(outlet_bcs) % 3):
        axs[max_row-1,2-i].remove()

    for i, bc_name in enumerate(outlet_bcs):

        row = int(i/3)
        col = i % 3

        axs[row, col].fill_between((7,13), (1/6, 1/6), color="tab:orange", label="Prior")
        y, x, _ = axs[row, col].hist(particles[:,i], weights=weights, bins=200, density=True, range=(7,13), label="Posterior")
        axs[row, col].axvline(x=gt[i], color="red", linewidth=1)
        axs[row, col].axvline(x=map[i], color="black", linewidth=1, linestyle="--", dashes=(5, 5))
        axs[row, col].set_xlabel(labels[i])
        axs[row, col].set_ylabel(f"PDF")
        axs[row, col].set_xlim((7, 13))

        axs[row, col].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axs[row, col].xaxis.set_major_locator(MaxNLocator(integer=True))
        if map[i] < 10:
            axs[row, col].annotate(rf"$\mu={mean[i]:.1f}$"+"\n"+rf"$\sigma={std[i]:.2f}$", (0.7, 0.5), xycoords="axes fraction", fontsize=11)
        else:
            axs[row, col].annotate(rf"$\mu={mean[i]:.1f}$"+"\n"+rf"$\sigma={std[i]:.2f}$", (0.05, 0.5), xycoords="axes fraction", fontsize=11)

    # img = plt.imread(model_img)
    # newax = fig.add_axes(model_image_loc[model_name], anchor='NE', zorder=1)
    # newax.imshow(img)
    # newax.axis('off')

    # fig.align_ylabels(axs[:, 0])
    # fig.align_ylabels(axs[:, 1])
    # fig.align_ylabels(axs[:, 2])
    fig.tight_layout(pad=0.2)
    fig.legend(loc="lower right")
    fig.savefig(os.path.join(target_folder, f"posterior_{model_name}.png"))
    fig.savefig(os.path.join(target_folder, f"posterior_{model_name}.svg"))