"""
Tests vector_heat_1d_2pts
"""
import numpy as np
import os

from pymgrit.induction_machine.grid_transfer_machine import GridTransferMachine
from pymgrit.induction_machine.vector_machine import VectorMachine


def test_grid_transfer_machine_constructor():
    """
    constructor
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    grid_transfer = GridTransferMachine(coarse_grid='im_3kW_4k', fine_grid='im_3kW_17k', path_meshes=path)

    assert len(grid_transfer.transfer_data) == 12
    assert len(grid_transfer.transfer_data['vtxInner']) == 6432
    assert len(grid_transfer.transfer_data['wtsInner']) == 6432
    assert len(grid_transfer.transfer_data['vtxOuter']) == 6615
    assert len(grid_transfer.transfer_data['vtxOuter']) == 6615
    assert grid_transfer.transfer_data['addBoundInner'] == 663
    assert grid_transfer.transfer_data['addBoundOuter'] == 76
    assert grid_transfer.transfer_data['sizeLvlStop'] == 17496
    assert grid_transfer.transfer_data['sizeLvlStart'] == 4449
    assert len(grid_transfer.transfer_data['mappingInner']) == 2208
    assert len(grid_transfer.transfer_data['mappingOuter']) == 2241
    assert len(grid_transfer.transfer_data['mappingInnerNew']) == 6432
    assert len(grid_transfer.transfer_data['mappingOuterNew']) == 6615


def test_grid_transfer_restriction():
    """
    restriction
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    grid_transfer = GridTransferMachine(coarse_grid='im_3kW_4k', fine_grid='im_3kW_17k', path_meshes=path)
    vec = VectorMachine(u_front_size=2, u_middle_size=4449, u_back_size=2)
    res_vec = grid_transfer.restriction(u=vec)

    np.testing.assert_equal(True, isinstance(res_vec, VectorMachine))


def test_grid_transfer_interpolation():
    """
    interpolation
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    grid_transfer = GridTransferMachine(coarse_grid='im_3kW_4k', fine_grid='im_3kW_17k', path_meshes=path)
    vec = VectorMachine(u_front_size=2, u_middle_size=4449, u_back_size=2)
    res_vec = grid_transfer.interpolation(u=vec)

    np.testing.assert_equal(True, isinstance(res_vec, VectorMachine))
