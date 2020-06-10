"""
Grid Transfer class for the induction
machine model "im_3kW". (https://gitlab.onelab.info/doc/models/-/wikis/Electric-machines)

For more details see: https://arxiv.org/abs/1912.03106

Interpolation: standard finite element interpolation
Restriction: injection
"""

import copy

import numpy as np

from pymgrit.core.grid_transfer import GridTransfer
from pymgrit.induction_machine.vector_machine import VectorMachine
from pymgrit.induction_machine.helper import check_version, compute_data, interpolation_factors, compute_mesh_transfer


class GridTransferMachine(GridTransfer):
    """
    Grid Transfer class for the induction machine im_3kW
    Interpolation: standard finite element interpolation
    Restriction: injection
    """

    def __init__(self, coarse_grid, fine_grid, path_meshes):
        """
        Constructor. Compute the transfer grid between both given grids in both directions.

        :param coarse_grid: Coarse grid name
        :param fine_grid:  Fine grid name
        :param path_meshes: Path to meshes
        """
        super().__init__()
        data_coarse_pre = path_meshes + coarse_grid + '.pre'
        data_coarse_msh = path_meshes + coarse_grid + '.msh'
        check_version(msh_file=data_coarse_msh)
        data_coarse = compute_data(data_coarse_pre, data_coarse_msh, 0)

        data_fine_pre = path_meshes + fine_grid + '.pre'
        data_fine_msh = path_meshes + fine_grid + '.msh'
        check_version(msh_file=data_fine_msh)
        data_fine = compute_data(data_fine_pre, data_fine_msh, len(data_coarse['corToUn']))

        self.transfer_data = interpolation_factors(data_coarse=data_coarse, data_fine=data_fine)

    def restriction(self, u: VectorMachine) -> VectorMachine:
        """
        Restriction

        :param u: approximate solution vector
        :return: input solution vector u restricted to a coarse grid
        """
        ret = copy.deepcopy(u)
        ret.u_middle = ret.u_middle[:self.transfer_data['sizeLvlStart']]
        ret.u_middle_size = self.transfer_data['sizeLvlStart']
        return ret

    def interpolation(self, u: VectorMachine) -> VectorMachine:
        """
        Interpolation

        :param u: approximate solution vector
        :return: input solution vector u interpolated to a fine grid
        """
        ret = copy.deepcopy(u)

        new_middle = np.zeros(self.transfer_data['sizeLvlStop'] - self.transfer_data['sizeLvlStart'])

        new_u_inner = compute_mesh_transfer(
            u.u_middle[self.transfer_data['mappingInner']], self.transfer_data['vtxInner'],
            self.transfer_data['wtsInner'], self.transfer_data['addBoundInner'], 0)

        new_u_outer = compute_mesh_transfer(
            u.u_middle[self.transfer_data['mappingOuter']], self.transfer_data['vtxOuter'],
            self.transfer_data['wtsOuter'], self.transfer_data['addBoundOuter'], 0)
        new_middle[:len(u.u_middle)] = u.u_middle
        new_middle[self.transfer_data['mappingInnerNew']] = new_u_inner
        new_middle[self.transfer_data['mappingOuterNew']] = new_u_outer
        ret.u_middle = np.append(ret.u_middle, new_middle)
        ret.u_middle_size = len(ret.u_middle)
        return ret
