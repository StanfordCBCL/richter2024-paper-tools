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

width = global_config["figure.figsize"][0] * 1.4

target_folder = os.path.join(this_file_dir, "build", "anatomies")


theta_id = [11, 14, 1]

fig, axs = plt.subplots(1, 3, figsize=(width, width*1/3), sharex="col")


os.makedirs(target_folder, exist_ok=True)

for iter, model_name in enumerate(["0104_0001"]):

    folder = f"/Volumes/richter/projects/{model_name}/ParameterEstimation/multi_fidelity_lm_calibration2"

    three_to_zero =  sorted([f for f in os.listdir(folder) if f.endswith("adaptive_three_d_simulation")], key=lambda x: int(x.split("_")[0]))

    project = SimVascularProject(f"/Volumes/richter/projects/{model_name}")

    threed_results_file = f"/Volumes/richter/72_centerline_results_martin/{model_name}.vtp"

    ground_truth = os.path.join(folder, three_to_zero[-1], "result.vtp")

    threed_result_handler = CenterlineHandler.from_file(threed_results_file)
    ground_truth_handler = CenterlineHandler.from_file(ground_truth)


    geometric_input_file =f"/Volumes/richter/projects/{model_name}/ROMSimulations/{model_name}/solver_0d.in"
    zerod_handler = SvZeroDSolverInputHandler.from_file(geometric_input_file)

    result_branch_data, times_3d = taskutils.map_centerline_result_to_0d_2(
        zerod_handler,
        threed_result_handler,
        project["3d_simulation_input"],
        threed_result_handler,
        padding=True,
    )

    gt_branch_data, times_3d_gt = taskutils.map_centerline_result_to_0d_2(
        zerod_handler,
        ground_truth_handler,
        project["3d_simulation_input"],
        ground_truth_handler,
        padding=True,
    )

    bc_map = zerod_handler.vessel_to_bc_map


    cycle = matplotlib.rcParams["axes.prop_cycle"]
    colors = ["blue", "red", "tab:orange"]

    bc_name = "INFLOW"
    bc = bc_map[bc_name]

    branch_name = bc["name"]
    branch_id, seg_id = branch_name.split("_")
    branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

    result_threed_pressure = taskutils.cgs_pressure_to_mmgh(result_branch_data[branch_id][seg_id][bc["pressure"]])
    result_threed_flow = taskutils.cgs_flow_to_lmin(result_branch_data[branch_id][seg_id][bc["flow"]])

    gt_threed_pressure = taskutils.cgs_pressure_to_mmgh(gt_branch_data[branch_id][seg_id][bc["pressure"]])
    gt_threed_flow = taskutils.cgs_flow_to_lmin(gt_branch_data[branch_id][seg_id][bc["flow"]])

    axs[iter].plot(times_3d_gt, gt_threed_pressure, color="red", label="Ground Truth")
    axs[iter].plot(times_3d, result_threed_pressure, color="black", linestyle="--", dashes=(5, 5), label="Calibrated")
    axs[iter].title.set_text("Pressure at inlet")

    # axs[row, col].axvline(x=gt[i], color="red", linewidth=1)
    #  axs[row, col].axvline(x=map[i], color="black", linewidth=1, linestyle="--", dashes=(5, 5))

    # axs[1, iter].plot(times_3d_gt, gt_threed_flow, color="red")
    # axs[1, iter].plot(times_3d, result_threed_flow, color="black", linestyle="--", dashes=(5, 5))

    axs[iter].set_xlabel("Time [s]")
    axs[iter].set_ylabel("Pressure [mmHg]")
    # axs[1, iter].set_ylabel("Flow [mmHg]")

fig.tight_layout()
# fig.align_ylabels(axs[:, 0])
# fig.align_ylabels(axs[:, 1])
# fig.align_ylabels(axs[:, 2])
axs[2].remove()
fig.legend()
fig.savefig(os.path.join(target_folder, f"curve_{model_name}.png"))
fig.savefig(os.path.join(target_folder, f"curve_{model_name}.svg"))