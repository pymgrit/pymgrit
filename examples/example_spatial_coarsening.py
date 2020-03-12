"""
Apply three-level MGRIT V-cycles to solve the 1D heat equation.
Additional spatial coarsening is used to reduce the size of the space problem on the coarser levels.
"""

import numpy as np

from pymgrit.heat.heat_1d import Heat1D  # 1D Heat equation problem
from pymgrit.heat.heat_1d import VectorHeat1D  # 1D Heat equation vector class
from pymgrit.core.mgrit import Mgrit  # MGRIT solver
from pymgrit.core.grid_transfer import GridTransfer  # Parent grid transfer class
from pymgrit.core.grid_transfer_copy import GridTransferCopy  # Copy transfer class


# Create a class for the grid transfer between spatial grids.
# The class mus inherit from PyMGRIT's core GridTransfer class.
class GridTransferHeat(GridTransfer):
    """
    Grid Transfer for the Heat Equation.
    Interpolation: Linear interpolation
    Restriction: Full weighted
    """

    def __init__(self):
        """
        Constructor.
        :rtype: object
        """
        super(GridTransferHeat, self).__init__()

    # Specify restriction operator
    def restriction(self, u: VectorHeat1D) -> VectorHeat1D:
        """
        Restrict u using full weighting.

        Note: The 1d heat equation example is with homogeneous Dirichlet BCs in space.
              The Heat1D vector class stores only the non boundary points.
        :param u: VectorHeat1D
        :rtype: VectorHeat1D
        """
        # Get the non boundary points
        sol = u.get_values()

        # Create array
        ret_array = np.zeros(int((len(sol) - 1) / 2))

        # Full weighting
        for i in range(len(ret_array)):
            ret_array[i] = sol[2 * i] * 1 / 4 + sol[2 * i + 1] * 1 / 2 + sol[2 * i + 2] * 1 / 4

        # Create and return a VectorHeat1D object
        ret = VectorHeat1D(len(ret_array))
        ret.set_values(ret_array)
        return ret

    # Specify interpolation operator
    def interpolation(self, u: VectorHeat1D) -> VectorHeat1D:
        """
        Interpolate u using linear interpolation

        Note: The 1d heat equation example is with homogeneous Dirichlet BCs in space.
              The Heat1D vector class stores only the non boundary points.
        :param u: VectorHeat1D
        :rtype: VectorHeat1D
        """
        # Get the non boundary points
        sol = u.get_values()

        # Create array
        ret_array = np.zeros(int(len(sol) * 2 + 1))

        # Linear interpolation
        for i in range(len(sol)):
            ret_array[i * 2] += 1 / 2 * sol[i]
            ret_array[i * 2 + 1] += sol[i]
            ret_array[i * 2 + 2] += 1 / 2 * sol[i]

        # Create and return a VectorHeat1D object
        ret = VectorHeat1D(len(ret_array))
        ret.set_values(ret_array)
        return ret


def main():
    heat0 = Heat1D(x_start=0, x_end=2, nx=2 ** 4 + 1, a=1, t_start=0, t_stop=2, nt=2 ** 7 + 1)
    heat1 = Heat1D(x_start=0, x_end=2, nx=2 ** 3 + 1, a=1, t_interval=heat0.t[::2])
    heat2 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat1.t[::2])
    heat3 = Heat1D(x_start=0, x_end=2, nx=2 ** 2 + 1, a=1, t_interval=heat2.t[::2])

    problem = [heat0, heat1, heat2, heat3]

    # Specify a list of grid transfer operators of length (#level - 1)
    # Using the new class GridTransferHeat to apply spatial coarsening on the first two levels
    # Using the PyMGRIT's core class GridTransferCopy on the last level (no spatial coarsening)
    transfer = [GridTransferHeat(), GridTransferHeat(), GridTransferCopy()]

    # Setup MGRIT solver with problem and transfer
    mgrit = Mgrit(problem=problem, transfer=transfer)

    info = mgrit.solve()


if __name__ == '__main__':
    main()
