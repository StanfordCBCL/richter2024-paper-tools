import os
import click
from svsuperestimator.main import run_from_config
import shutil

# python batch_bloodvessel.py /Volumes/richter/final_data/projects /Volumes/richter/final_data/input_0d /Volumes/richter/final_data/centerlines/ /Volumes/richter/final_data/centerline_results model_calibration_nov5


@click.command()
@click.argument("model_folder")
@click.argument("zerod_folder")
@click.argument("centerline_folder")
@click.argument("solution_folder")
@click.argument("tag")
def main(model_folder, zerod_folder, centerline_folder, solution_folder, tag):
    saw_0081 = False
    for project_name in [
        f.removesuffix(".vtp")
        for f in os.listdir(model_folder)
        if not f.startswith(".")
    ]:
        if project_name == "0081_0001":
            saw_0081 = True
        if not saw_0081:
            continue
        project_folder = os.path.join(model_folder, project_name)
        calibrated_file = os.path.join(
            project_folder, "ParameterEstimation", tag, "solver_0d.in"
        )
        report = os.path.join(
            project_folder, "ParameterEstimation", tag, "report.html"
        )
        # calibrated_file_target = os.path.join(
        #     "/Volumes/richter/final_data/model_calibration_nov5",
        #     f"{project_name}_0d.in",
        # )
        # report_target = os.path.join(
        #     "/Volumes/richter/final_data/model_calibration_nov5",
        #     f"{project_name}.html",
        # )
        # shutil.copyfile(calibrated_file, calibrated_file_target)
        # shutil.copyfile(report, report_target)
        # continue

        # if os.path.exists(os.path.join(project_folder, "ParameterEstimation", tag, "report.html")):
        #     print("Skipping", project_name)
        #     continue

        config = {
            "project": project_folder,
            "file_registry": {
                "centerline": os.path.join(
                    os.path.abspath(centerline_folder), f"{project_name}.vtp"
                )
            },
            "tasks": {
                "model_calibration_least_squares": {
                    "name": tag,
                    "zerod_config_file": os.path.join(
                        zerod_folder, f"{project_name}_0d.in"
                    ),
                    "threed_solution_file": os.path.join(
                        solution_folder, f"{project_name}.vtp"
                    ),
                    "maximum_iterations": 100,
                    "overwrite": True,
                }
            },
        }
        if project_name == "0081_0001":
            config["tasks"]["model_calibration_least_squares"]["calibrate_stenosis_coefficient"] = False
        run_from_config(config)

if __name__ == "__main__":
    main()
