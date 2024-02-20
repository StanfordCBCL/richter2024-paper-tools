
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


for noise_level, tuning_name in zip([0.1, 0.3, 0.5], ["multi_fidelity_january2024_0104_0001", "multi_fidelity_february_2024_more_noise", "multi_fidelity_february2024_even_more_noise"]):

    model_name = "0104_0001"

    # project = SimVascularProject(f"/Volumes/richter/final_data/projects/{model_name}")
    zerod_handler = SvZeroDSolverInputHandler.from_file(f"/Volumes/richter/final_data/input_0d/{model_name}_0d.in")

    metrics_all = {}

    windkessel_tunings = [f"/Volumes/richter/final_data/posterior_plots/{tuning_name}/5_windkessel_tuning"]

    # for wk_tuning in windkessel_tunings:

    with open(os.path.join(windkessel_tunings[-1], "results.json")) as ff:
        data = json.load(ff)
    
    with open(os.path.join(windkessel_tunings[-1], "taskdata.json")) as ff:
        taskdata = json.load(ff)

    gt = np.array(data["metrics"]["ground_truth"])
    # map = np.array(data["metrics"]["maximum_a_posteriori"])

    weights = np.array(taskdata["weights"][-1]).flatten()
    particles = np.array(taskdata["particles"][-1])
    
    # ---- sort by weight and select top n_top particles ----
    n_top = 5000
    weight_sort = np.argsort(weights)[::-1][:n_top]

    weights = weights[weight_sort]
    particles = particles[weight_sort]

    mean = np.average(particles, weights=weights, axis=0)
    covmat = np.cov(particles.T, aweights=weights)
    std = np.array([covmat[i][i] ** 0.5 for i in range(covmat.shape[0])])

    outlet_bcs = list(zerod_handler.outlet_boundary_conditions)

    labels = [r"$\theta^{" + f"({i+1})" + "}$" for i in range(len(outlet_bcs))]

    data = pd.DataFrame(columns = labels, data=particles)
    # print(data)
    # raise SystemExit

    colormap_name = "Blues"

    cmap = matplotlib.colormaps[colormap_name]

    g = sns.PairGrid(data)
    g.figure.set_size_inches(width*2/3,width*2/3)
    # plt.title(r"$f_{\sigma}=" + f"{int(noise_level*100)}$")
    g.map_upper(sns.scatterplot, hue=weights, linewidth=0)
    g.map_lower(sns.kdeplot, weights=weights, fill=True)
    g.map_diag(sns.kdeplot, weights=weights, fill=True, linewidth=1)

    # ---- Adjusting Axis Ranges ----
    global_xmin = 7
    global_xmax = 13
    global_ymin = 7
    global_ymax = 13
    g.set(xlim=(global_xmin, global_xmax), ylim=(global_ymin, global_ymax))

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

    g.savefig(os.path.join(target_folder, "plots", f"posterior_{model_name}_{int(noise_level*100)}.png"))