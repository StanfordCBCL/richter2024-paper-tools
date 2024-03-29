import os

import numpy as np
from rich import print
from svsuperestimator import reader
from svsuperestimator.tasks import taskutils

threed_results_folder = "/Volumes/richter/final_data/centerline_results"

zerod_input_folder = "/Volumes/richter/final_data/input_0d"

model_names = ["0104_0001", "0140_2001", "0080_0001"]

target_folder = "/Volumes/richter/final_data/multi_fidelity_calibration/input"


for model_name in model_names:

    print(model_name)
    project = reader.SimVascularProject(
        os.path.join("/Volumes/richter/projects", model_name)
    )

    # Target format: p_in_min, p_in_max, q_out_mean_1, q_out_mean_2, ...

    zerod_handler = reader.SvZeroDSolverInputHandler.from_file(os.path.join(zerod_input_folder, model_name + "_0d.in"))

    result_handler = reader.CenterlineHandler.from_file(
        os.path.join(threed_results_folder, model_name + ".vtp")
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

    theta_obs = np.log(
        [
            bc["bc_values"]["Rd"] + bc["bc_values"]["Rp"]
            for bc in outlet_bcs.values()
        ]
    ).tolist()

    config = f"""project: /scratch/users/richter7/data/projects/{model_name}
global:
    num_procs: 48
slurm:
    partition: amarsden
    python-path: ~/miniconda3/envs/estimator/bin/python
file_registry:
    centerline: /scratch/users/richter7/data/centerlines/{model_name}.vtp
    centerline_path: /scratch/users/richter7/data/centerlines/{model_name}.vtp
    0d_simulation_input_path: /scratch/users/richter7/data/input_0d/{model_name}_0d.in
    0d_simulation_input: /scratch/users/richter7/data/input_0d/{model_name}_0d.in
tasks:
    multi_fidelity_tuning:
        name: multi_fidelity_january2024
        theta_obs: {theta_obs}
        y_obs: {y_obs}
        smc_num_particles: 10000
        smc_num_rejuvenation_steps: 2
        smc_resampling_threshold: 0.5
        smc_noise_factor: 0.1
        three_d_max_asymptotic_error: 0.01
        svpre_executable: /home/users/richter7/svsolver/build/svSolver-build/bin/svpre
        svsolver_executable: /home/users/richter7/svsolver/build/svSolver-build/bin/svsolver
        svpost_executable: /home/users/richter7/svsolver/build/svSolver-build/bin/svpost
        svslicer_executable: /home/users/richter7/svSlicer/Release/svslicer
    """

    with open(os.path.join(target_folder, f"{model_name}_estimator.yaml"), "w") as ff:
        ff.write(config)
