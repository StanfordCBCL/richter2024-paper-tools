import json
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import MaxNLocator
from rich import print
from svsuperestimator import reader
from svsuperestimator.reader import (CenterlineHandler, SimVascularProject,
                                     SvZeroDSolverInputHandler)
from svsuperestimator.tasks import taskutils

this_file_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_file_dir, "config.json")) as ff:
    global_config = json.load(ff)
matplotlib.rcParams.update(global_config)

target_folder = os.path.join(this_file_dir, "build", "anatomies")

os.makedirs(target_folder, exist_ok=True)

width = global_config["figure.figsize"][0]

metrics = {
    "0104_0001": {},
}

for model_name in ["0104_0001"]:

    # model_img = f"six_anatomies/bc_descs/{model_name}.png"

    project = SimVascularProject(f"/Volumes/richter/projects/{model_name}")
    zerod_handler = project["0d_simulation_input"]

    folder = f"/Volumes/richter/projects/{model_name}/ParameterEstimation/multi_fidelity_lm_calibration2"

    config_file = f"/Volumes/richter/projects/{model_name}/ParameterEstimation/multi_fidelity_lm_calibration2/config.yaml"

    windkessel_tunings = sorted([f for f in os.listdir(folder) if f.endswith("windkessel_tuning")], key=lambda x: int(x.split("_")[0]))[:7]

    centerline_results =  sorted([f for f in os.listdir(folder) if f.endswith("adaptive_three_d_simulation")], key=lambda x: int(x.split("_")[0]))[:7]

    data = []
    number_of_0d_sims = 0
    for wk_tuning in windkessel_tunings:
        wk_folder = os.path.join(folder, wk_tuning)


        with open(os.path.join(wk_folder, "taskdata.json")) as ff:
            taskdata = json.load(ff)

        num_smc_steps = np.array(taskdata["particles"]).shape[0]
        print(num_smc_steps)

        # num_steps = len([f for f in os.listdir(wk_folder) if f.startswith("results_")])
        number_of_0d_sims += 30000 + (num_smc_steps-1) * 20000
        
        try:
            with open(os.path.join(folder, wk_tuning, "results.json")) as ff:
                data.append(json.load(ff))
        except:
            break
    
    metrics[model_name]["num_zerod_eval"] = number_of_0d_sims
    metrics[model_name]["num_threed_eval"] = len(centerline_results)

    with open(config_file) as ff:
        config = yaml.safe_load(ff)


    y_obs_target = np.array(config["tasks"]["multi_fidelity_tuning"]["y_obs"])

    outlet_bcs = zerod_handler.outlet_boundary_conditions

    gt = np.array(data[0]["metrics"]["ground_truth"])
    gt_exp = np.exp(gt)

    map_erros = []
    map_std =[]
    for d in data:
        map_erros.append(np.mean(d["metrics"]["maximum_a_posteriori_error"]))
        map_std.append(np.std(d["metrics"]["maximum_a_posteriori_error"]))

    y_obs_erros = []
    y_obs_stds = []

    for centerline_name in centerline_results:

        cl_file = os.path.join(folder, centerline_name, "result.vtp")

        try:
            result_handler = reader.CenterlineHandler.from_file(
                os.path.join(cl_file)
            )
        except FileNotFoundError:
            result_handler = reader.CenterlineHandler.from_file(
                os.path.join(cl_file)
            )

        data_raw, times = taskutils.map_centerline_result_to_0d_2(
            zerod_handler,
            result_handler,# project["3d_simulation_input"],
            project["3d_simulation_input"],
            result_handler,
            padding=True
        )

        data = {}

        for branch_id, branch in data_raw.items():
            for seg_id, seg in branch.items():
                data[f"branch{branch_id}_seg{seg_id}"] = seg

        outlet_bcs = zerod_handler.outlet_boundary_conditions
        bc_map = zerod_handler.vessel_to_bc_map

        y_obs = []
        pressure_in = data[bc_map["INFLOW"]["name"]][bc_map["INFLOW"]["pressure"]]
        y_obs.append(np.amin(pressure_in))
        y_obs.append(np.amax(pressure_in))

        for bc_name in outlet_bcs:

            flow_out = data[bc_map[bc_name]["name"]][bc_map[bc_name]["flow"]]

            y_obs.append(np.mean(flow_out))

        y_obs_erros.append(np.mean(np.abs(np.array(y_obs)-y_obs_target)/y_obs_target))
        y_obs_stds.append(np.std(np.abs(np.array(y_obs)-y_obs_target)/y_obs_target))
    

    metrics[model_name]["map_err"] = map_erros[-1]
    metrics[model_name]["map_std"] = map_std[-1]
    metrics[model_name]["yobs_err"] = y_obs_erros[-1]
    metrics[model_name]["yobs_std"] = y_obs_stds[-1]


    fig, axs = plt.subplots(1, 2, figsize=(width, width*1/2))


    axs[0].plot(np.arange(1, len(map_erros)+1), np.array(map_erros)*100, color="tab:blue")
    axs[1].plot(np.arange(1, len(map_erros)+1), np.array(y_obs_erros)*100, color="tab:orange")

    # axs[0].set_yscale("log")
    # axs[1].set_yscale("log")

    # tick_y = [10**x for x in range(-1, 2)]
    # axs[0].set_yticks(tick_y, [f"{y} %" for y in tick_y])
    # axs[1].set_yticks(tick_y, [f"{y} %" for y in tick_y])

    # axs[0].grid()
    # axs[1].grid()

    axs[0].set_xlabel("$n$")
    axs[0].set_ylabel(r"Error of $\boldsymbol{\theta}_{\mathrm{MAP}, n+1} [\%]$")
    axs[0].title.set_text("MAP error")
    axs[1].set_xlabel("$n$")
    axs[1].set_ylabel(r"Error of $\boldsymbol{y}_{\mathrm{3D}, n+1} [\%]$")
    axs[1].title.set_text("3D results error")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))


    fig.tight_layout()
    fig.savefig(os.path.join(target_folder, f"errors_{model_name}.png"))
    fig.savefig(os.path.join(target_folder, f"errors_{model_name}.svg"))


print(metrics)