
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from svsuperestimator.reader import SvZeroDSolverInputHandler
from matplotlib import style, rcParams
import seaborn as sns
import pandas as pd
import matplotlib

# Make the plot



this_file_dir = os.path.abspath(os.path.dirname(__file__))

style.use(os.path.join(this_file_dir, "matplotlibrc"))

width = rcParams["figure.figsize"][0]
height = rcParams["figure.figsize"][1]

target_folder = "/Volumes/richter/final_data/posterior_plots"

os.makedirs(target_folder, exist_ok=True)


for model_name in ["0104_0001"]:

    # project = SimVascularProject(f"/Volumes/richter/final_data/projects/{model_name}")
    zerod_handler = SvZeroDSolverInputHandler.from_file(f"/Volumes/richter/final_data/input_0d/{model_name}_0d.in")

    metrics_all = {}

    windkessel_tunings = [f"/Volumes/richter/final_data/posterior_plots/multi_fidelity_january2024_{model_name}/5_windkessel_tuning"]

    # for wk_tuning in windkessel_tunings:

    with open(os.path.join(windkessel_tunings[-1], "results.json")) as ff:
        data = json.load(ff)
    
    with open(os.path.join(windkessel_tunings[-1], "taskdata.json")) as ff:
        taskdata = json.load(ff)

    gt = np.array(data["metrics"]["ground_truth"])
    # map = np.array(data["metrics"]["maximum_a_posteriori"])

    particles = np.array(taskdata["particles"][-1])
    weights = np.array(taskdata["weights"][-1]).flatten()

    # print(particles.shape)
    # print(weights.shape)
    # raise SystemExit

    # print(np.sum(weights))

    # print(np.concatenate([particles, weights.reshape(-1, 1)], axis=1).shape)
    # print(particles.shape)

    mean = np.average(particles, weights=weights, axis=0)
    covmat = np.cov(particles.T, aweights=weights)
    std = np.array([covmat[i][i] ** 0.5 for i in range(covmat.shape[0])])

    outlet_bcs = list(zerod_handler.outlet_boundary_conditions)

    labels = [r"$\theta^{" + f"({i+1})" + "}$" for i in range(len(outlet_bcs))]

    max_row = int(len(outlet_bcs) / 3) if len(outlet_bcs) % 3 == 0 else int(len(outlet_bcs) / 3) + 1

    data = pd.DataFrame(columns = labels, data=particles)
    # print(data)
    # raise SystemExit

    colormap_name = "Blues"

    cmap = matplotlib.colormaps[colormap_name]

    g = sns.PairGrid(data) 
    g.map_upper(sns.scatterplot, hue=weights, palette=colormap_name, linewidth=0)
    g.map_lower(sns.kdeplot, weights=weights, cmap=colormap_name, fill=True)
    g.map_diag(sns.histplot, legend=False, bins=50, weights=weights, color=cmap(1.0), linewidth=0)

    # x_min = np.amin(particles)
    # x_max = np.amax(particles)
    # offset = 0.1 * (x_max-x_min)
    # x_min-=offset
    # x_max+=offset
    # for i in range(particles.shape[1]):
    #     for j in range(particles.shape[1]):
    #         if i == j:
    #             g.axes[i, j].set_xlim((x_min, x_max))
    #             continue
    #         g.axes[i, j].set_xlim((x_min, x_max))
    #         g.axes[i, j].set_ylim((x_min, x_max))

    g.savefig(os.path.join(target_folder, "plots", f"posterior_{model_name}.png"))