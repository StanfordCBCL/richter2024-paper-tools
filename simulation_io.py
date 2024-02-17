#!/usr/bin/env python

import os
import vtk

from collections import OrderedDict

from vtk.util.numpy_support import vtk_to_numpy as v2n


def read_geo(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh

    Returns:
        vtk reader, point data, cell data
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader


def get_caps(f_outlet, f_centerline):
    """
    Map outlet names to centerline branch id
    Args:
        f_outlet: ordered list of outlet names (created during centerline extraction)
        f_centerline: centerline geometry (.vtp)

    Returns:
        dictionary {cap name: BranchId}
    """
    caps = OrderedDict()
    caps['inflow'] = 0

    # read ordered outlet names from file
    outlet_names = []
    with open(f_outlet) as file:
        for line in file:
            outlet_names += line.splitlines()

    # read centerline
    cent = read_geo(f_centerline).GetOutput()
    if not cent.GetPointData().HasArray('BranchId'):
        raise RuntimeError('centerline branch extraction failed')
    branch_id = v2n(cent.GetPointData().GetArray('BranchId'))

    # find outlets and store outlet name and BranchId
    ids = vtk.vtkIdList()
    i_outlet = 0

    # loop all centerline points
    for i in range(1, cent.GetNumberOfPoints()):
        cent.GetPointCells(i, ids)

        # check if cap
        if ids.GetNumberOfIds() == 1:
            # this works since the points are numbered according to the order of outlets
            caps[outlet_names[i_outlet]] = branch_id[i]
            i_outlet += 1
    
    assert len(outlet_names) == i_outlet, "outlet number mismatch"
    return caps

