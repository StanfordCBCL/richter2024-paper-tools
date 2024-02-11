
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from svsuperestimator.reader import SvZeroDSolverInputHandler
from matplotlib import style, rcParams
import seaborn as sns
import pandas as pd
import matplotlib

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# Make the plot



this_file_dir = os.path.abspath(os.path.dirname(__file__))

style.use(os.path.join(this_file_dir, "matplotlibrc"))

width = rcParams["figure.figsize"][0]
height = rcParams["figure.figsize"][1]

target_folder = "/Volumes/richter/final_data/posterior_plots"

os.makedirs(target_folder, exist_ok=True)

# # Sample data (1D for simplicity)
# data = np.random.normal(0, 1, size=1000)
# data = data.reshape(-1, 1)  # Reshape for scikit-learn

# # Range of bandwidths to try
# bandwidths = np.linspace(0.1, 1.0, 50)

# # Grid search with cross-validation
# grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#                     {'bandwidth': bandwidths},
#                     cv=5)  # 5-fold cross-validation
# grid.fit(data)

# # Best bandwidth
# best_bandwidth = grid.best_params_['bandwidth']
# print(f"Best bandwidth: {best_bandwidth}")


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
    logpost = np.array(taskdata["logpost"][-1]).flatten()

    # print(np.min(weights))
    # print(np.max(weights))
    # raise SystemExit
    # print(particles.shape)
    # print(weights.shape)
    # raise SystemExit

    # print(np.sum(weights))

    # print(np.concatenate([particles, weights.reshape(-1, 1)], axis=1).shape)
    # print(particles.shape)

    # mean = np.average(particles, weights=weights, axis=0)
    # covmat = np.cov(particles.T, aweights=weights)
    # std = np.array([covmat[i][i] ** 0.5 for i in range(covmat.shape[0])])

    outlet_bcs = list(zerod_handler.outlet_boundary_conditions)

    labels = [r"$\theta^{" + f"({i+1})" + "}$" for i in range(len(outlet_bcs))]

    max_row = int(len(outlet_bcs) / 3) if len(outlet_bcs) % 3 == 0 else int(len(outlet_bcs) / 3) + 1

    data = pd.DataFrame(columns = labels, data=particles)
    # print(data)
    # raise SystemExit
    
    bandwidths = []
    bins = []
    
    # raise SystemExit

    for i in range(particles.shape[1]):

        particles_i = particles[:, i].reshape(-1, 1)[::10]
        weights_i = weights[::10]

        weights_i /= np.sum(weights_i)

        min_part = particles_i.min()
        max_part = particles_i.max()

        range_parti = max_part - min_part

        # print(min_part, max_part)
        # raise SystemExit
    
        # Range of bandwidths to try
        trial_bandwidths = np.linspace(0.001*range_parti, 0.01*range_parti, 50)
        # trial_bandwidths = np.logspace(-3, -1, num=100, base=10)
        # print(trial_bandwidths)
        # raise SystemExit

        # Grid search with cross-validation
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': trial_bandwidths},
                            cv=20)  # 5-fold cross-validation
        grid.fit(particles_i, sample_weight=weights_i)

        # Best bandwidth
        best_bandwidth = grid.best_params_['bandwidth']

        bandwidths.append(best_bandwidth)
        bins.append(int(range_parti / best_bandwidth))

        print(f"{i} | Minimum: {0.01*range_parti:.4f} | Maximum: {0.1*range_parti:.4f} | Chose: {best_bandwidth:.4f}")

    print(bins)
    print(bandwidths)
    bins = [376, 521, 844, 844, 644]
    bandwidths = [0.0027232863927803014, 0.0011590493487084904, 0.0010891642577173452, 0.0011921364631954012, 0.0015853595525720824]

    colormap_name = "Blues"

    cmap = matplotlib.colormaps[colormap_name]


    # def my_bandwidth_func(dataset):

    #     particles = dataset.dataset

    #     particles_1 = particles[1,:]
    #     particles_2 = particles[1,:]

    #     print("Input", particles.shape)
    #     raise SystemExit


    g = sns.PairGrid(data) 
    g.map_upper(sns.scatterplot, hue=weights, palette=colormap_name, linewidth=0)
    g.map_lower(sns.kdeplot, weights=weights, cmap=colormap_name, fill=True, bw_method=0.2)
    g.map_diag(sns.histplot, legend=False, bins=844, color=cmap(1.0), linewidth=0)

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