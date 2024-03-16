import json
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style, rcParams
import multiprocessing
from tempfile import TemporaryDirectory
import seaborn as sns

import numpy as np
import pandas as pd
from rich import box, print
from rich.table import Table
from scipy.stats import pearsonr
from svsuperestimator.reader import *
from svsuperestimator.reader import utils as readutils
from svsuperestimator.tasks import taskutils
import pysvzerod
from svsuperestimator.main import run_from_config
import shutil

import matplotlib.patheffects as path_effects

import utils


def run_calibration(project_folder, centerline, zerod_file, threed_solution, calibrated_file_target):
    calibrated_folder = os.path.join(project_folder, "ParameterEstimation", "tmp")
    calibrated_file = os.path.join(calibrated_folder, "solver_0d.in")

    if os.path.exists(calibrated_folder):
        shutil.rmtree(calibrated_folder)

    config = {
        "project": project_folder,
        "file_registry": {
            "centerline": centerline
        },
        "tasks":{
            "model_calibration_least_squares": {
                "name": "tmp",
                "zerod_config_file": zerod_file,
                "threed_solution_file": threed_solution,
                "maximum_iterations": 100,
                "overwrite": True,
                "post_proc": False,
                "report_html": False,
                "calibrate_stenosis_coefficient": False # Set only for 0080_0001
            }
        }
    }
    run_from_config(config)
    shutil.copyfile(calibrated_file, calibrated_file_target)
    shutil.rmtree(calibrated_folder)

def update_boundary_conditions(
    project_path, variation_config, three_d_result_file, zerod_file, centerline_file, target_zerod_file
):
    project = SimVascularProject(project_path, {"0d_simulation_input": zerod_file, "centerline": centerline_file})
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

    branch_data_3d, times = taskutils.map_centerline_result_to_0d_3(
        zerod_handler,
        CenterlineHandler.from_file(centerline_file),
        handler3d,
        threed_result_handler,
    )

    inflow_data = ""
    inflow_raw = branch_data_3d[0][0]["flow_in"]
    new_inflow = taskutils.refine_with_cubic_spline(inflow_raw, 1000)
    new_times = np.linspace(times[0], times[-1], 1000)
    for time, flow in zip(new_times, new_inflow):
        inflow_data += f"{time:.18e} {-flow:.18e}\n"

    handler_inflow = SvSolverInflowHandler(inflow_data)

    new_params = utils.map_3d_boundary_conditions_to_0d(
        project, handler_rcr, handler3d, handlermesh, handler_inflow
    )

    utils.update_zero_d_boundary_conditions(zerod_handler, new_params)
    zerod_handler.to_file(target_zerod_file)

def get_metrics(project_path, zerod_file, zerod_opt_file, centerline_file, three_d_result_file):

    project = SimVascularProject(project_path, {"0d_simulation_input": zerod_file, "centerline": centerline_file})

    handler3d = project["3d_simulation_input"]
    threed_result_handler = CenterlineHandler.from_file(three_d_result_file, padding=True)

    zerod_handler = SvZeroDSolverInputHandler.from_file(zerod_file)
    zerod_opt_handler = SvZeroDSolverInputHandler.from_file(zerod_opt_file)

    branch_data_3d, times = taskutils.map_centerline_result_to_0d_3(
        zerod_handler,
        CenterlineHandler.from_file(centerline_file),
        handler3d,
        threed_result_handler,
    )

    taskutils.set_initial_condition(zerod_handler, branch_data_3d, times)
    taskutils.set_initial_condition(zerod_opt_handler, branch_data_3d, times)

    zerod_handler.update_simparams(
        last_cycle_only=True, num_cycles=1, steady_initial=False, max_nliter=1000
    )
    zerod_opt_handler.update_simparams(
        last_cycle_only=True, num_cycles=1, steady_initial=False, max_nliter=1000
    )

    result_0d = pysvzerod.simulate(zerod_handler.data)
    result_0d_opt = pysvzerod.simulate(zerod_opt_handler.data)

    result_0d_sys_caps = utils.get_systolic_pressure_and_flow_at_caps_0d(
        zerod_handler, result_0d
    )
    result_0d_sys_caps_opt = utils.get_systolic_pressure_and_flow_at_caps_0d(
        zerod_opt_handler, result_0d_opt
    )

    result_3d_sys_caps = utils.get_systolic_pressure_and_flow_at_caps_3d(
        zerod_handler, branch_data_3d
    )

    return {
        "pressure_3d": result_3d_sys_caps[0],
        "flow_3d": result_3d_sys_caps[1],
        "pressure_0d": result_0d_sys_caps[0],
        "flow_0d": result_0d_sys_caps[1],
        "pressure_0d_opt": result_0d_sys_caps_opt[0],
        "flow_0d_opt": result_0d_sys_caps_opt[1],
    }

def multip_function(args):
    
    project_path, i, variation_config, threed_result_file, zerod_file, zerod_opt_file, centerline_file = args

    with TemporaryDirectory() as tmpdir:

        zerod_file_i = os.path.join(tmpdir, "input0.json")
        zerod_opt_file_i = os.path.join(tmpdir, "input1.json")

        update_boundary_conditions(project_path, variation_config, threed_result_file, zerod_file, centerline_file, zerod_file_i)
        update_boundary_conditions(project_path, variation_config, threed_result_file, zerod_opt_file, centerline_file, zerod_opt_file_i)

        result = get_metrics(project_path, zerod_file_i, zerod_opt_file_i, centerline_file, threed_result_file)
    print(f"\tFinished with {i}")
    return result, f"variation_{i}"

def add_median_labels(ax: plt.Axes, fmt: str = ".1f") -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    lines_per_box = len(lines) // len(boxes)
    i = 0
    for median in lines[4::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        if i >= 6:
            text = ax.text(x+0.1, y, f'{value:{fmt}}', ha='left', va='center',
                        #    fontweight='bold',
                        color='black',
                        fontsize=7)
        else:
            text = ax.text(x-0.11, y, f'{value:{fmt}}', ha='right', va='center',
                        #    fontweight='bold',
                        color='black',
                        fontsize=7)
        i +=1
        # create median-colored border around white text for contrast
        # text.set_path_effects([
        #     path_effects.Stroke(linewidth=1, foreground=median.get_color()),
        #     path_effects.Normal(),
        # ])

def main():

    this_file_dir = os.path.abspath(os.path.dirname(__file__))

    style.use(os.path.join(this_file_dir, "matplotlibrc"))

    width = rcParams["figure.figsize"][0]

    luca_folder = "/Volumes/richter/final_data/lucas_result_january/vtps"
    project_folder = "/Volumes/richter/final_data/projects"

    data_set_info = "/Volumes/richter/final_data/lucas_result_january/vtps/dataset_info.json"

    with open(data_set_info) as ff:
        data_config = json.load(ff)

    target_folder = os.path.join("/Volumes/richter/final_data/3d_0d_calibration")

    os.makedirs(target_folder, exist_ok=True)

    df_target = os.path.join(target_folder, "errors.csv")

    # for patient in ["0080_0001"]: #"0104_0001"# Model "0080_0001" as resistance BCs
    #     project_path = os.path.join(project_folder, patient)

    #     # zerod_opt_file = os.path.join("/Volumes/richter/final_data/results_lm_72_2", patient + "_0d.in")
    #     zerod_file = os.path.join("//Volumes/richter/final_data/input_0d", patient + "_0d.in")
    #     centerline_file = os.path.join("/Volumes/richter/final_data/centerlines", patient + ".vtp")

    #     data = {}

    #     for k in range(50):
    #         print(f"Starting variation {k}")
    #         tag = f"{patient}/variation_{k}"

    #         os.makedirs(os.path.join(target_folder, patient), exist_ok=True)

    #         zerod_file_i = os.path.join(target_folder, tag + "_geo.json")
    #         zerod_opt_file_i = os.path.join(target_folder, tag + "_cal.json")

    #         variation_config = data_config[f"s{patient}.{k}" + ".vtp"]
    #         threed_result_file= os.path.join(luca_folder, f"s{patient}.{k}" + ".vtp")

    #         update_boundary_conditions(project_path, variation_config, threed_result_file, zerod_file, centerline_file, zerod_file_i)
    #         run_calibration(project_path, centerline_file, zerod_file_i, threed_result_file, zerod_opt_file_i)

    #         with multiprocessing.Pool(10) as pool:
    #             result = pool.map(multip_function, [(project_path, i,  data_config[f"s{patient}.{i}" + ".vtp"], os.path.join(luca_folder, f"s{patient}.{i}" + ".vtp"), zerod_file_i, zerod_opt_file_i, centerline_file) for i in range(50)])
    
    #         data = {t: vardata for vardata, t in result}

    #         with open(os.path.join(target_folder, tag + "_data.json"), "w") as ff:
    #             json.dump(data, ff, indent=4)

    # headers = ["Model", "Calibrated", "Mean systolic pressure error at caps [\%]", "Mean systolic flow error at caps [\%]"]

    # df = pd.DataFrame(columns=headers)
    # idx = 0
    # for patient in ["0104_0001", "0140_2001", "0080_0001"]:

    #     with open(os.path.join(target_folder, f"{patient}/variation_0" + "_data.json")) as ff:
    #         data =json.load(ff)

    #     # validation error
    #     pressure_3d = np.array([vardata["pressure_3d"] for key, vardata in data.items()])
    #     flow_3d = np.array([vardata["flow_3d"] for key, vardata in data.items()])
    #     pressure_0d = np.array([vardata["pressure_0d"] for key, vardata in data.items()])
    #     flow_0d = np.array([vardata["flow_0d"] for key, vardata in data.items()])

    #     pres_error = np.mean(np.abs((pressure_0d-pressure_3d)/pressure_3d), axis=1) * 100
    #     flow_error = np.mean(np.abs((flow_0d-flow_3d)/flow_3d), axis=1) * 100

    #     for i in range(len(pres_error)):
    #         df.loc[idx] = [patient, "Geometric\nN=50", pres_error[i], flow_error[i]]
    #         idx+=1

    #     for k in range(50):

    #         tag = f"{patient}/variation_{k}"

    #         with open(os.path.join(target_folder, tag + "_data.json")) as ff:
    #             data =json.load(ff)
                
    #         var_id = lambda x: int(x.split("_")[1])

    #         # validation error
    #         pressure_3d = np.array([vardata["pressure_3d"] for key, vardata in data.items() if var_id(key) != k])
    #         flow_3d = np.array([vardata["flow_3d"] for key, vardata in data.items() if var_id(key) != k])
    #         pressure_0d_opt = np.array([vardata["pressure_0d_opt"] for key, vardata in data.items() if var_id(key) != k])
    #         flow_0d_opt = np.array([vardata["flow_0d_opt"] for key, vardata in data.items() if var_id(key) != k])

    #         # training error
    #         pressure_3d_train = np.array([vardata["pressure_3d"] for key, vardata in data.items() if var_id(key) == k])
    #         flow_3d_train = np.array([vardata["flow_3d"] for key, vardata in data.items() if var_id(key) == k])
    #         # pressure_0d_train = np.array([vardata["pressure_0d"] for key, vardata in data.items() if var_id(key) == k]).flatten()
    #         # flow_0d_train = np.array([vardata["flow_0d"] for key, vardata in data.items() if var_id(key) == k]).flatten()
    #         pressure_0d_opt_train = np.array([vardata["pressure_0d_opt"] for key, vardata in data.items() if var_id(key) == k])
    #         flow_0d_opt_train = np.array([vardata["flow_0d_opt"] for key, vardata in data.items() if var_id(key) == k])

    #         pres_error_opt = np.mean(np.abs((pressure_0d_opt-pressure_3d)/pressure_3d), axis=1) * 100
    #         flow_error_opt = np.mean(np.abs((flow_0d_opt-flow_3d)/flow_3d), axis=1) * 100

    #         pres_error_opt_train = np.mean(np.abs((pressure_0d_opt_train-pressure_3d_train)/pressure_3d_train), axis=1) * 100
    #         flow_error_opt_train = np.mean(np.abs((flow_0d_opt_train-flow_3d_train)/flow_3d_train), axis=1) * 100

    #         for i in range(len(pres_error_opt_train)):
    #             df.loc[idx] = [patient, "Optimized (training)\nN=50", pres_error_opt_train[i], flow_error_opt_train[i]]
    #             idx+=1
    #         for i in range(len(pres_error_opt)):
    #             df.loc[idx] = [patient, "Optimized (validation)\nN=49x50", pres_error_opt[i], flow_error_opt[i]]
    #             idx+=1

    # df.to_csv(df_target)

    df = pd.read_csv(df_target)

    fig, axs = plt.subplots(1, 2, figsize=[width, width*0.65], sharey=True)

    sns.boxplot(df, x="Model", y="Mean systolic pressure error at caps [\%]", hue="Calibrated", ax=axs[0], palette="Greens", linewidth=0.5, width=.5, fliersize=0, saturation=1)

    add_median_labels(axs[0])

    sns.boxplot(df, x="Model", y="Mean systolic flow error at caps [\%]", hue="Calibrated", ax=axs[1], palette="Greens", linewidth=0.5, width=.5, fliersize=0, saturation=1)

    add_median_labels(axs[1])

    axs[0].set_title("Pressure", y=1.02)
    axs[0].set_ylim([0, 17.5])
    axs[1].set_title("Flow", y=1.02)
    axs[1].set_ylim([0, 17.5])
    axs[0].set_ylabel("Mean systolic error at caps [\%]")
    axs[0].grid(axis='y')
    axs[1].grid(axis='y')
    axs[0].set_xlabel("")
    axs[1].set_xlabel("")

    axs[0].spines["top"].set_visible(True)
    axs[0].spines["right"].set_visible(True)
    axs[1].spines["top"].set_visible(True)
    axs[1].spines["right"].set_visible(True)

    axs[0].legend(loc='upper right', bbox_to_anchor=(1.93, -0.29), ncol=3)
    axs[1].get_legend().remove()

    opt_0d_name = r"$\mathfrak{M}_\text{0D}(\boldsymbol{\theta}_\text{VMR}, \hat{\boldsymbol{\alpha}})$"
    geo_0d_name = r"$\mathfrak{M}_\text{0D}(\boldsymbol{\theta}_\text{VMR}, \boldsymbol{\alpha}_\text{b})$"

    new_labels = [f"{geo_0d_name}\nN=50", f"{opt_0d_name} (training)\nN=50", f"{opt_0d_name} (validation)\nN=49x50"]
    for t, l in zip(axs[0].get_legend().texts, new_labels):
        t.set_text(l)

    locations = [
        [0.11,0.11,0.05,0.11],
        [0.26,0.11,0.05,0.11],
        [0.415,0.11,0.05,0.11],
    ]
    for i, model in enumerate(["0104_0001", "0140_2001", "0080_0001"]):

        model_img = f"/Users/jakobrichter/code/paper-tools/data/pictures/{model}.png"
        img = plt.imread(model_img)
        newax = fig.add_axes(locations[i], anchor='NE', zorder=1)
        newax.imshow(img)
        newax.axis('off')
    for loc in locations:
        loc[0] += 0.465
    for i, model in enumerate(["0104_0001", "0140_2001", "0080_0001"]):

        model_img = f"/Users/jakobrichter/code/paper-tools/data/pictures/{model}.png"
        img = plt.imread(model_img)
        newax = fig.add_axes(locations[i], anchor='NE', zorder=1)
        newax.imshow(img)
        newax.axis('off')

    fig.subplots_adjust(bottom=0.28, left=0.08, right=0.98, top=0.9, wspace=0.07)

    # fig.tight_layout()
    fig.savefig(os.path.join(target_folder, "errors.png"))


if __name__ == "__main__":
    main()