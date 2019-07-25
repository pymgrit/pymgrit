from mpi4py import MPI
from mgrit import mgrit_fas as solver
from cable_voltage_driven import cable_voltage_driven
from cable_voltage_driven import grid_transfer_copy
import logging

if __name__ == '__main__':
    cable_0 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=129)
    cable_1 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=33)
    cable_2 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=5)
    problem = [cable_0, cable_1, cable_2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, grid_transfer=transfer, nested_iteration=True, it=5,debug_lvl=logging.INFO)
    k = mgrit.solve()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print('a')

