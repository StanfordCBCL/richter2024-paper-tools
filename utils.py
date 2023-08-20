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
from typing import Sequence
import plotly.express as px

# from svzerodsolver import runnercpp


def map_3d_boundary_conditions_to_0d(
    project, handler_rcr, handler3d, handlermesh, handler_inflow
):
    face_ids = handlermesh.boundary_centers

    mapped_data = readutils.get_0d_element_coordinates(project)

    bc_ids = list(face_ids.keys())
    bc_ids_coords = list(face_ids.values())

    bc_mapping = {}
    for bc_name, bc_coord in mapped_data.items():
        if not bc_name.startswith("branch"):
            index = np.argmin(np.linalg.norm(bc_ids_coords - bc_coord, axis=1))
            bc_mapping[bc_name] = bc_ids[index]

    threed_bc = handler_rcr.get_rcr_data()
    try:
        surface_ids = handler3d.rcr_surface_ids
    except AttributeError:
        surface_ids = handler3d.r_surface_ids
    for i, bc in enumerate(threed_bc):
        surface_id = surface_ids[i]

        for bc_name, bc_id in bc_mapping.items():
            if bc_id == surface_id:
                bc_mapping[bc_name] = bc
    bc_mapping["INFLOW"] = handler_inflow.get_inflow_data()
    return bc_mapping


def update_zero_d_boundary_conditions(zerod_handler, new_params):
    for bc_name, bc_config in zerod_handler.boundary_conditions.items():
        # print(bc_config["bc_values"])

        for param_name in bc_config["bc_values"].keys():
            if param_name in new_params[bc_name]:
                if (
                    isinstance(new_params[bc_name][param_name], Sequence)
                    and (len(new_params[bc_name][param_name]) == 2)
                    and (
                        new_params[bc_name][param_name][0]
                        == new_params[bc_name][param_name][1]
                    )
                ):
                    bc_config["bc_values"][param_name] = new_params[bc_name][
                        param_name
                    ][0]
                else:
                    bc_config["bc_values"][param_name] = new_params[bc_name][param_name]
            else:
                bc_config["bc_values"]["R"] = new_params[bc_name]["Rd"]
        
    #     print(bc_config["bc_values"])
    #     print("\n\n")
    # raise SystemExit




def get_systolic_pressure_and_flow_at_caps_0d(zerod_handler, result_0d):
    bc_map = zerod_handler.vessel_to_bc_map

    pressures = []
    flows = []

    for config in bc_map.values():
        vessel_name = config["name"]

        branch_id, seg_id = vessel_name.split("_")
        branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

        pressure_id = config["pressure"]
        flow_id = config["flow"]

        vessel_result_0d = result_0d[result_0d.name == vessel_name]

        # if vessel_name == "branch0_seg0":

        #     px.line(vessel_result_0d[pressure_id]).show()
        #     px.line(vessel_result_0d[flow_id]).show()

        # Collect 0d results
        pressures.append(np.amax(vessel_result_0d[pressure_id]))
        flows.append(np.amax(vessel_result_0d[flow_id]))

    return pressures, flows


def get_systolic_pressure_and_flow_at_caps_3d(zerod_handler, branch_data):
    bc_map = zerod_handler.vessel_to_bc_map

    pressures = []
    flows = []

    for config in bc_map.values():
        vessel_name = config["name"]

        branch_id, seg_id = vessel_name.split("_")
        branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

        segment = branch_data[branch_id][seg_id]

        pressure_id = config["pressure"]
        flow_id = config["flow"]

        # if vessel_name == "branch0_seg0":

        #     px.line(segment[pressure_id]).show()
        #     px.line(segment[flow_id]).show()

        # Collect 3d results
        pressures.append(np.amax(segment[pressure_id]))
        flows.append(np.amax(segment[flow_id]))

    return pressures, flows
