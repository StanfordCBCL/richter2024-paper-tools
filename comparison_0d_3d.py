import json
import os
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from rich import box, print
from rich.table import Table
from scipy.stats import pearsonr
from svsuperestimator.reader import *
from svsuperestimator.reader import utils as readutils
from svsuperestimator.tasks import taskutils
from svzerodsolver import runnercpp

import utils


def get_metrics(
    project_path, variation_config, three_d_result_file, zerod_file, zerod_opt_file
):
    project = SimVascularProject(project_path)
    handler3d = project["3d_simulation_input"]
    handler_rcr = SvSolverRcrHandler("")

    try:
        surface_ids = handler3d.rcr_surface_ids
    except AttributeError:
        surface_ids = handler3d.r_surface_ids
    boundary_centers = project["mesh"].boundary_centers
    centers = np.array([boundary_centers[idx] for idx in surface_ids])
    rcr_data = [None] * len(surface_ids)
    for key, value in variation_config.items():
        if isinstance(value, dict) and "RCR" in value:
            coord = value["point"]
            index = np.argmin(np.linalg.norm(centers-coord, axis=1))
            rcr_data[index] = dict(Rp=value["RCR"][0], C=value["RCR"][1], Rd=value["RCR"][2], Pd=[0.0, 0.0], t=[0.0, 1.0])
        if isinstance(value, dict) and "RP" in value:
            coord = value["point"]
            index = np.argmin(np.linalg.norm(centers-coord, axis=1))
            rcr_data[index] = dict(Rp=0.0, C=0.0, Rd=value["RP"][0], Pd=[value["RP"][1], value["RP"][1]], t=[0.0, 1.0])

    handler_rcr.set_rcr_data(rcr_data)

    handlermesh = project["mesh"]
    threed_result_handler = CenterlineHandler.from_file(three_d_result_file, padding=True)

    zerod_handler = SvZeroDSolverInputHandler.from_file(zerod_file)
    zerod_opt_handler = SvZeroDSolverInputHandler.from_file(zerod_opt_file)

    branch_data_3d, times = taskutils.map_centerline_result_to_0d_3(
        zerod_handler,
        project["centerline"],
        handler3d,
        threed_result_handler,
    )

    inflow_data = ""
    inflow_raw = branch_data_3d[0][0]["flow_in"]
    new_inflow = taskutils.refine_with_cubic_spline(inflow_raw, 1000)
    new_times = np.linspace(times[0], times[-1], 1000)
    for time, flow in zip(new_times, new_inflow):
        inflow_data += f"{time:.18e} {-flow:.18e}\n"

    # import plotly.express as px

    # px.line(x=times, y=inflow_raw).show()

    # raise SystemExit

    handler_inflow = SvSolverInflowHandler(inflow_data)

    new_params = utils.map_3d_boundary_conditions_to_0d(
        project, handler_rcr, handler3d, handlermesh, handler_inflow
    )

    # print(new_params)
    # px.line(x=new_params["INFLOW"]["t"], y=new_params["INFLOW"]["Q"]).show()
    # raise SystemExit

    utils.update_zero_d_boundary_conditions(zerod_handler, new_params)
    utils.update_zero_d_boundary_conditions(zerod_opt_handler, new_params)

    taskutils.set_initial_condition(zerod_handler, branch_data_3d, times)
    taskutils.set_initial_condition(zerod_opt_handler, branch_data_3d, times)

    zerod_handler.update_simparams(
        last_cycle_only=True, #num_cycles=1, steady_initial=False
    )
    zerod_opt_handler.update_simparams(
        last_cycle_only=True, #num_cycles=1, steady_initial=False
    )

    result_0d = runnercpp.run_from_config(zerod_handler.data)
    result_0d_opt = runnercpp.run_from_config(zerod_opt_handler.data)

    result_0d_sys_caps = utils.get_systolic_pressure_and_flow_at_caps_0d(
        zerod_handler, result_0d
    )
    result_0d_sys_caps_opt = utils.get_systolic_pressure_and_flow_at_caps_0d(
        zerod_opt_handler, result_0d_opt
    )

    result_3d_sys_caps = utils.get_systolic_pressure_and_flow_at_caps_3d(
        zerod_handler, branch_data_3d
    )

    return result_3d_sys_caps, result_0d_sys_caps, result_0d_sys_caps_opt


def main():

    this_file_dir = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_file_dir, "config.json")) as ff:
        global_config = json.load(ff)
    matplotlib.rcParams.update(global_config)

    width = global_config["figure.figsize"][0]

    luca_folder = "/Volumes/richter/lucas_result_january/vtps"
    project_folder = "/Volumes/richter/projects"

    data_set_info = "/Volumes/richter/lucas_result_january/vtps/dataset_info.json"

    with open(data_set_info) as ff:
        data_config = json.load(ff)

    for patient in ["0080_0001", "0104_0001", "0140_2001"]: #"0104_0001"# Model "0080_0001" as resistance BCs
        project_path = os.path.join(project_folder, patient)

        # zerod_opt_file = os.path.join("/Users/stanford/results_lm_72", patient + "_0d.in")
        zerod_opt_file = os.path.join("/Volumes/richter/projects", patient, "ParameterEstimation", "calibration_least_squares", "solver_0d.in")
        zerod_file = os.path.join("/Volumes/richter/projects", patient, "ROMSimulations", patient, "solver_0d.in")

        data = {}

        for i in range(50):
            var_name = f"s{patient}.{i}"
            print("Running ", patient, " ", var_name)

            threed_result_file = os.path.join(luca_folder, var_name + ".vtp")

            variation_config = data_config[var_name + ".vtp"]

            metrics = get_metrics(project_path, variation_config, threed_result_file, zerod_file, zerod_opt_file)

            data[f"variation_{i}"] = {
                "pressure_3d": metrics[0][0],
                "flow_3d": metrics[0][1],
                "pressure_0d": metrics[1][0],
                "flow_0d": metrics[1][1],
                "pressure_0d_opt": metrics[2][0],
                "flow_0d_opt": metrics[2][1],
            }
        
        pressure_3d = np.array([vardata["pressure_3d"] for vardata in data.values()]).flatten()
        flow_3d = np.array([vardata["flow_3d"] for vardata in data.values()]).flatten()
        pressure_0d = np.array([vardata["pressure_0d"] for vardata in data.values()]).flatten()
        flow_0d = np.array([vardata["flow_0d"] for vardata in data.values()]).flatten()
        pressure_0d_opt = np.array([vardata["pressure_0d_opt"] for vardata in data.values()]).flatten()
        flow_0d_opt = np.array([vardata["flow_0d_opt"] for vardata in data.values()]).flatten()

        pres_coef = np.corrcoef(pressure_3d, pressure_0d)[0][1]
        pres_coef_opt = np.corrcoef(pressure_3d, pressure_0d_opt)[0][1]
        flow_coef = np.corrcoef(flow_3d, flow_0d)[0][1]
        flow_coef_opt = np.corrcoef(flow_3d, flow_0d_opt)[0][1]
        
        pres_error = np.mean(np.abs((pressure_0d-pressure_3d)/pressure_3d)) * 100
        pres_error_opt = np.mean(np.abs((pressure_0d_opt-pressure_3d)/pressure_3d)) * 100
        flow_error = np.mean(np.abs((flow_0d-flow_3d)/flow_3d)) * 100
        flow_error_opt = np.mean(np.abs((flow_0d_opt-flow_3d)/flow_3d)) * 100

        fig, axs = plt.subplots(2, 2, figsize=[width, width])

        axs[0][0].text(0.1, 0.8, rf"$\rho={pres_coef:.2f}$", transform=axs[0][0].transAxes, fontsize=11)
        axs[0][1].text(0.1, 0.8, rf"$\rho={pres_coef_opt:.2f}$", transform=axs[0][1].transAxes, fontsize=11)
        axs[1][0].text(0.1, 0.8, rf"$\rho={flow_coef:.2f}$", transform=axs[1][0].transAxes, fontsize=11)
        axs[1][1].text(0.1, 0.8, rf"$\rho={flow_coef_opt:.2f}$", transform=axs[1][1].transAxes, fontsize=11)

        axs[0][0].text(0.1, 0.7, rf"$\bar\varepsilon={pres_error:.1f}\%$", transform=axs[0][0].transAxes, fontsize=11)
        axs[0][1].text(0.1, 0.7, rf"$\bar\varepsilon={pres_error_opt:.1f}\%$", transform=axs[0][1].transAxes, fontsize=11)
        axs[1][0].text(0.1, 0.7, rf"$\bar\varepsilon={flow_error:.1f}\%$", transform=axs[1][0].transAxes, fontsize=11)
        axs[1][1].text(0.1, 0.7, rf"$\bar\varepsilon={flow_error_opt:.1f}\%$", transform=axs[1][1].transAxes, fontsize=11)

        for i, suffix in enumerate(["", "_opt"]):
            max_pres = -9e99
            min_pres = 9e99
            for var_name, var_data in data.items():
                max_pres = max(
                    max(var_data["pressure_3d"]),
                    max(var_data["pressure_0d"+suffix]),
                    max_pres,
                )
                min_pres = min(
                    min(var_data["pressure_3d"]),
                    min(var_data["pressure_0d"+suffix]),
                    min_pres,
                )

                axs[0, i].scatter(
                    taskutils.cgs_pressure_to_mmgh(var_data["pressure_3d"]),
                    taskutils.cgs_pressure_to_mmgh(var_data["pressure_0d"+suffix]),
                    3,
                    # color="black"
                )
            prange = taskutils.cgs_pressure_to_mmgh([min_pres - 0.5e4, max_pres + 0.5e4])
            axs[0, i].set_xlim(prange)
            axs[0, i].set_ylim(prange)
            axs[0, i].plot(prange, prange, "--", color="black", alpha=0.7)
            axs[0, i].set_xlabel("3D systolic pressure [mmHg]")
            if i == 0:
                axs[0, i].set_ylabel("0D systolic pressure [mmHg]")
                axs[0, i].title.set_text("Geometric model")
            else:
                axs[0, i].title.set_text("Calibrated model")

        for i, suffix in enumerate(["", "_opt"]):
            max_flow = -9e99
            min_flow = 9e99
            for var_name, var_data in data.items():
                max_flow = max(
                    max(var_data["flow_3d"]),
                    max(var_data["flow_0d"+suffix]),
                    max_flow,
                )
                min_flow = min(
                    min(var_data["flow_3d"]),
                    min(var_data["flow_0d"+suffix]),
                    min_flow,
                )
                axs[1, i].scatter(
                    taskutils.cgs_flow_to_lmin(var_data["flow_3d"]),
                    taskutils.cgs_flow_to_lmin(var_data["flow_0d"+suffix]),
                    3,
                    # color="black"
                )
            frange = taskutils.cgs_flow_to_lmin([min_flow - 20, max_flow + 20])
            axs[1, i].set_xlim(frange)
            axs[1, i].set_ylim(frange)
            axs[1, i].plot(frange, frange, "--", color="black", alpha=0.7)
            axs[1, i].set_xlabel("3D systolic flow [l/min]")
            if i == 0:
                axs[1, i].set_ylabel("0D systolic flow [l/min]")

        target_folder = os.path.join(this_file_dir, "build", "comparison_0d_3d")

        os.makedirs(target_folder, exist_ok=True)

        fig.tight_layout()
        fig.savefig(os.path.join(target_folder, f"{patient}.svg"))
        fig.savefig(os.path.join(target_folder, f"{patient}.png"))

if __name__ == "__main__":
    main()