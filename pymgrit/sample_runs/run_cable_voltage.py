import logging

from pymgrit.core import mgrit as solver
from pymgrit.core import grid_transfer_copy
from pymgrit.cable_voltage_driven import cable_voltage_driven


def main():
    cable_0 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=129)
    cable_1 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=33)
    cable_2 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=5)
    problem = [cable_0, cable_1, cable_2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.Mgrit(problem=problem, transfer=transfer, nested_iteration=True, it=5, logging_lvl=logging.INFO)
    return mgrit.solve()


if __name__ == '__main__':
    main()
