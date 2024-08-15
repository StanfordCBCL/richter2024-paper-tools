from svsuperestimator.tasks.three_d_simulation import AdaptiveThreeDSimulation
from svsuperestimator.reader import SimVascularProject

centerline = "/Volumes/richter/final_data/centerlines/0104_0001.vtp"

project = SimVascularProject(
    "/Volumes/richter/0104_0001",
    registry_override={
        "0d_simulation_input": "/Volumes/richter/final_data/input_0d/0104_0001_0d.in",
        "0d_simulation_input_path": "/Volumes/richter/final_data/input_0d/0104_0001_0d.in",
        "centerline": centerline,
    },
)
task = AdaptiveThreeDSimulation(
    project,
    config={
        "core_run": False,
        "zerod_config_file": "/Volumes/richter/final_data/input_0d/0104_0001_0d.in",
        "overwrite": True,
    },
    parent_folder="/Volumes/richter/0104_0001/ParameterEstimation/grid_sampling_3d_february2024_0",
)
task.run()
