"""
Apply four-level MGRIT V-cycles to solve the 1D heat equation.

Additional spatial coarsening is used to reduce the size of the spatial problems on coarse levels.
"""

import numpy as np

from pymgrit.heat.heat_1d import Heat1D  # 1D Heat equation problem
from pymgrit.heat.heat_1d import VectorHeat1D  # 1D Heat equation vector class
from pymgrit.core.mgrit import Mgrit  # MGRIT solver
from pymgrit.core.grid_transfer import GridTransfer  # Parent grid transfer class
from pymgrit.core.grid_transfer_copy import GridTransferCopy  # Copy transfer class


# Create class for the grid transfer between spatial grids.
# Note: The class must inherit from PyMGRIT's core GridTransfer class.
class GridTransferHeat(GridTransfer):
    """
    Grid Transfer class for the Heat Equation.
    Interpolation: Linear interpolation
    Restriction: Full weighting
    """

    def __init__(self):
        """
        Constructor.
        :rtype: GridTransferHeat object
        """
        super(GridTransferHeat, self).__init__()

    # Define restriction operator
    def restriction(self, u: VectorHeat1D) -> VectorHeat1D:
        """
        Restrict input vector u using standard full weighting restriction.

        Note: In the 1d heat equation example, we consider homogeneous Dirichlet BCs in space.
              The Heat1D vector class only stores interior points.
        :param u: approximate solution vector
        :return: input solution vector u restricted to a coarse grid
        """
        # Get values at interior points
        sol = u.get_values()

        # Create array for restricted values
        ret_array = np.zeros(int((len(sol) - 1) / 2))

        # Full weighting restriction
        for i in range(len(ret_array)):
            ret_array[i] = sol[2 * i] * 1 / 4 + sol[2 * i + 1] * 1 / 2 + sol[2 * i + 2] * 1 / 4

        # Create and return a VectorHeat1D object with the restricted values
        ret = VectorHeat1D(len(ret_array))
        ret.set_values(ret_array)
        return ret

    # Define interpolation operator
    def interpolation(self, u: VectorHeat1D) -> VectorHeat1D:
        """
        Interpolate input vector u using linear interpolation.

        Note: In the 1d heat equation example, we consider homogeneous Dirichlet BCs in space.
              The Heat1D vector class only stores interior points.
        :param u: approximate solution vector
        :return: input solution vector u interpolated to a fine grid
        """
        # Get values at interior points
        sol = u.get_values()

        # Create array for interpolated values
        ret_array = np.zeros(int(len(sol) * 2 + 1))

        # Linear interpolation
        for i in range(len(sol)):
            ret_array[i * 2] += 1 / 2 * sol[i]
            ret_array[i * 2 + 1] += sol[i]
            ret_array[i * 2 + 2] += 1 / 2 * sol[i]

        # Create and return a VectorHeat1D object with interpolated values
        ret = VectorHeat1D(len(ret_array))
        ret.set_values(ret_array)
        return ret


def main():
    # Construct a four-level multigrid hierarchy for the 1d heat example
    #   * use a coarsening factor of 2 in time on all levels
    #   * apply spatial coarsening by a factor of 2 on the first two levels
    heat0 = Heat1D(x_start=0, x_end=2, nx=2 ** 4 + 1, a=1, t_start=0, t_stop=2, nt=2 ** 7 + 1)
    heat1 = Heat1D(x_start=0, x_end=2, nx=2 ** 3 + 1, a=1, t_interval=heat0.t[::2])
    heat2 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat1.t[::2])
    heat3 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat2.t[::2])

    problem = [heat0, heat1, heat2, heat3]

    # Specify a list of grid transfer operators of length (#levels - 1) for the transfer between two consecutive levels
    #   * Use the new class GridTransferHeat to apply spatial coarsening for transfers between the first three levels
    #   * Use PyMGRIT's core class GridTransferCopy for the transfer between the last two levels (no spatial coarsening)
    transfer = [GridTransferHeat(), GridTransferHeat(), GridTransferCopy()]

    # Setup four-level MGRIT solver and solve the problem
    mgrit = Mgrit(problem=problem, transfer=transfer)

    info = mgrit.solve()


if __name__ == '__main__':
    main()
