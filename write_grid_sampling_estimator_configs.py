import os

target_folder = "/Volumes/richter/final_data/grid_sampling/sample_estimator_configs"

sample_configs = "/expanse/lustre/scratch/jrichter/temp_project/grid_sampling/sample_configs"

for filename in os.listdir("/Volumes/richter/final_data/grid_sampling/sample_configs"):

    index = int(filename.removesuffix(".json").split("_")[-1])
    sample_config = os.path.join(sample_configs, filename)

    model_name = "0104_0001"

    config = f"""project: /expanse/lustre/scratch/jrichter/temp_project/grid_sampling/projects/{model_name}
global:
    num_procs: 128
slurm:
    partition: compute
    python-path: ~/miniconda3/envs/estimator/bin/python
    nodes: 1
    mem: 64GB
    ntasks-per-node: 128
    account: TG-CTS130034
file_registry:
    centerline: /expanse/lustre/scratch/jrichter/temp_project/grid_sampling/centerlines/{model_name}.vtp
    centerline_path: /expanse/lustre/scratch/jrichter/temp_project/grid_sampling/centerlines/{model_name}.vtp
    0d_simulation_input_path: /expanse/lustre/scratch/jrichter/temp_project/grid_sampling/input_0d/{model_name}_0d.in
    0d_simulation_input: /expanse/lustre/scratch/jrichter/temp_project/grid_sampling/input_0d/{model_name}_0d.in
tasks:
    three_d_simulation_from_zero_d_config:
        name: grid_sampling_3d_february2024_{index}
        zerod_config_file: {sample_config}
        svpre_executable: /home/jrichter/svSolver/BuildWithMake/Bin/svpre.exe
        svsolver_executable: /home/jrichter/svSolver/BuildWithMake/Bin/svsolver-openmpi.exe
        svpost_executable: /home/jrichter/svSolver/BuildWithMake/Bin/svpost.exe
        svslicer_executable: /home/jrichter/svSlicer/Release/svslicer
    """

    with open(os.path.join(target_folder, f"sample_{index}.yaml"), "w") as ff:
        ff.write(config)
