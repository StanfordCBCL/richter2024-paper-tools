import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from click import style
from rich import box, print
from rich.table import Table
from scipy.stats import pearsonr
from svsuperestimator.reader import *
from svsuperestimator.reader import utils as readutils
from svsuperestimator.tasks import taskutils
from svzerodsolver import runnercpp

this_file_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_file_dir, "config.json")) as ff:
    global_config = json.load(ff)
matplotlib.rcParams.update(global_config)

width = global_config["figure.figsize"][0]

target_folder = os.path.join(this_file_dir, "build", "comparison_0d_3d")

model_img = "/Users/stanford/thesis-tools/model_img/0129_0000.png"

os.makedirs(target_folder, exist_ok=True)

project = SimVascularProject("/Volumes/richter/projects/0129_0000")

geometric_input_file = "/Volumes/richter/projects/0129_0000/ROMSimulations/0129_0000/solver_0d.in"
calibrated_input_file = "/Volumes/richter/projects/0129_0000/ParameterEstimation/zerod_three_comparison/solver_0d.in"
threed_results_file = "/Volumes/richter/72_centerline_results_martin/0129_0000.vtp"


geometric_zerod_handler = SvZeroDSolverInputHandler.from_file(geometric_input_file)
calibrated_zerod_handler = SvZeroDSolverInputHandler.from_file(calibrated_input_file)

threed_result_handler = CenterlineHandler.from_file(threed_results_file)



branch_data, times_3d = taskutils.map_centerline_result_to_0d_2(
    geometric_zerod_handler,
    project["centerline"],
    project["3d_simulation_input"],
    threed_result_handler,
    padding=True,
)

# taskutils.set_initial_condition(geometric_zerod_handler, branch_data)
# taskutils.set_initial_condition(calibrated_zerod_handler, branch_data)
geometric_zerod_handler.update_simparams(
    last_cycle_only=True
)
calibrated_zerod_handler.update_simparams(
    last_cycle_only=True
)

result_0d = runnercpp.run_from_config(geometric_zerod_handler.data)
result_0d_opt = runnercpp.run_from_config(calibrated_zerod_handler.data)


times_0d = np.array(result_0d[result_0d.name == "branch0_seg0"]["time"])

bc_map = geometric_zerod_handler.vessel_to_bc_map

fig, axs = plt.subplots(1, 7, figsize=[width*1.5, width * 0.3], sharex=True, sharey='row')

axs[6].remove()
# axs[1, 6].remove()


# img = plt.imread(model_img)
# newax = fig.add_axes([0.7,0.1,0.3,0.5], anchor='NE', zorder=1)
# newax.imshow(img)
# newax.axis('off')

titles = {
    "INFLOW": "AA", # Ascending aorta
    "RCR_0": "DA", # Descending aorta
    "RCR_4": "LS", # Left subclavian
    "RCR_1": "LCC", # Left common carotid
    "RCR_2": "RS", # Right subclavian
    "RCR_3": "RCC", # Right common carotid
}

handles = [None, None, None]

cycle = matplotlib.rcParams["axes.prop_cycle"]
colors = ["black", "red", "dodgerblue"]

for i, (bc_name, bc) in enumerate(bc_map.items()):

    branch_name = bc["name"]
    branch_id, seg_id = branch_name.split("_")
    branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

    subtitle = titles[bc_name]

    geom_pressure = taskutils.cgs_pressure_to_mmgh(np.array(result_0d[result_0d.name == bc["name"]][bc["pressure"]]))
    geom_flow = taskutils.cgs_flow_to_lmin(np.array(result_0d[result_0d.name == bc["name"]][bc["flow"]]))
    calib_pressure = taskutils.cgs_pressure_to_mmgh(np.array(result_0d_opt[result_0d_opt.name == bc["name"]][bc["pressure"]]))
    calib_flow = taskutils.cgs_flow_to_lmin(np.array(result_0d_opt[result_0d_opt.name == bc["name"]][bc["flow"]]))

    threed_pressure = taskutils.cgs_pressure_to_mmgh(branch_data[branch_id][seg_id][bc["pressure"]])
    threed_flow = taskutils.cgs_flow_to_lmin(branch_data[branch_id][seg_id][bc["flow"]])

    axs[i].title.set_text(subtitle)
    handles[0] = axs[i].plot(times_3d, threed_pressure, color=colors[0], label="3D", linewidth=1.0)[0]
    # axs[1, i].plot(times_3d, threed_flow, color=colors[0], linewidth=0.7)


    handles[1] = axs[i].plot(times_0d, geom_pressure, "--", color=colors[1], label="0D geo.", linewidth=1.5)[0]
    handles[2] = axs[i].plot(times_0d, calib_pressure, "--", dashes=(4, 4), color=colors[2], label="0D cal.", linewidth=1.5)[0]
    # axs[1, i].plot(times_0d, geom_flow,"--", color=colors[1], linewidth=0.7)
    # axs[1, i].plot(times_0d, calib_flow,"-.", color=colors[2], linewidth=0.7)

    axs[i].set_xlabel("Time [s]")
    axs[0].set_ylabel("Pressure [mmHg]")
    # axs[1, 0].set_ylabel("Flow [l/min]")

    axs[i].set_xlim(xmin=times_0d[0], xmax=times_0d[-1])
    # axs[1, i].set_xlim(xmin=times_0d[0], xmax=times_0d[-1])

    # axs[i].grid()
    axs[i].set_yticks(np.arange(70, 120, 20))
    # axs[1, i].set_yticks(np.arange(0, 30, 10))
    # axs[1, i].grid()

    # axs[1, i].set_xticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])


fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.98, 0.5))
# fig.align_ylabels(axs[0])
fig.tight_layout(pad=0.6)
fig.savefig(os.path.join(target_folder, f"time_plot_0129_0000.svg"))
fig.savefig(os.path.join(target_folder, f"time_plot_0129_0000.png"))