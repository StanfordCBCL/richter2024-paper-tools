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
from svzerodsolver import runnercpp

this_file_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_file_dir, "config.json")) as ff:
    global_config = json.load(ff)
matplotlib.rcParams.update(global_config)

target_folder = os.path.join(this_file_dir, "build", "anatomies")

os.makedirs(target_folder, exist_ok=True)

width = global_config["figure.figsize"][0]


all_times = {}

for model_name in ["0104_0001"]:

    project = SimVascularProject(f"/Volumes/richter/projects/{model_name}")
    zerod_handler = project["0d_simulation_input"]

    folder = f"/Volumes/richter/projects/{model_name}/ParameterEstimation/multi_fidelity_lm_calibration2"

    config_file = f"/Volumes/richter/projects/{model_name}/ParameterEstimation/multi_fidelity_lm_calibration2/config.yaml"

    windkessel_tunings = sorted([f for f in os.listdir(folder) if f.endswith("windkessel_tuning")], key=lambda x: int(x.split("_")[0]))

    print(windkessel_tunings)

    zerod_to_three = sorted([f for f in os.listdir(folder) if f.endswith("map_zero_d_result_to_three_d")], key=lambda x: int(x.split("_")[0]))

    print(zerod_to_three)

    threed_simulation = sorted([f for f in os.listdir(folder) if f.endswith("adaptive_three_d_simulation")], key=lambda x: int(x.split("_")[0]))

    print(threed_simulation)

    # three_to_zero =  sorted([f for f in os.listdir(folder) if f.endswith("map_three_d_result_on_centerline")], key=lambda x: int(x.split("_")[0]))

    # print(three_to_zero)

    bv_tuning = sorted([f for f in os.listdir(folder) if f.endswith("model_calibration_least_squares")], key=lambda x: int(x.split("_")[0]))

    print(bv_tuning)

    windkessel_tunings_time = 0.0

    times = {
        "windkessel_tunings": 0.0,
        "zerod_to_three": 0.0,
        "threed_simulation": 0.0,
        "bv_tuning": 0.0,
    }

    tasks = {
        "windkessel_tunings": windkessel_tunings,
        "zerod_to_three": zerod_to_three,
        "threed_simulation": threed_simulation,
        "bv_tuning": bv_tuning,
    }

    for name, task_folders in tasks.items():

        for taskname in task_folders:
            task_db = os.path.join(folder, taskname, "taskdata.json")

            try:
                with open(task_db) as ff:
                    data = json.load(ff)
            except:
                continue
            
            times[name] += data["core_runtime"]
    
    times["threed_simulation"] += times["zerod_to_three"]
    del times["zerod_to_three"]
    # times["bv_tuning"] += times["three_to_zero"]
    # del times["three_to_zero"]
    
    all_times[model_name] = times

print(all_times)

colors = ["#D91A1A", "#8F1D1E", "#462022"]

fig, ax = plt.subplots(figsize=(width*.8, width*.8))

times = list(all_times["0104_0001"].values())

print("Total runtime:", np.sum(times) / 3600 / 7)

percentages = np.array(times) / np.sum(times) * 100
ax.pie(list(times), labels=[f"BC calibration {percentages[0]:.1f}$\%$", f"3D simulation {percentages[1]:.1f}$\%$", f"0D optimization {percentages[2]:.1f}$\%$"])

fig.tight_layout()

fig.savefig(os.path.join(target_folder, f"runtimes.png"))
fig.savefig(os.path.join(target_folder, f"runtimes.svg"))

raise SystemExit

        
labels = ["0104_0001"]

fig, ax = plt.subplots(figsize=(width, width*0.3))

previous_values = np.array([0.0, 0.0, 0.0])


task_names = {
    "windkessel_tunings": "BC calibration",
    "threed_simulation": "3D simulation",
    "bv_tuning": "0D model calibration",
}

colors = ["#D91A1A", "#8F1D1E", "#462022"]

all_values = 0.0

print(all_times)

for model, runtimes in all_times.items():

    print(model)

for i, task_id in enumerate(times.keys()):
    print(task_id)

    values = np.array([all_times[l][task_id] for l in labels])/3600
    all_values += values

    ax.barh(labels, values, left=previous_values,
       label=task_names[task_id], height=0.8, color=colors[i])

    previous_values += values

ax.set_xlabel("Runtime [h]")

print(all_values)

fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.9))
fig.tight_layout()
fig.savefig(os.path.join(target_folder, f"runtimes.png"))
fig.savefig(os.path.join(target_folder, f"runtimes.svg"))

    


