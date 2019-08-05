from mgrit import mgrit as solver
from cable_voltage_driven import cable_voltage_driven
from cable_voltage_driven import grid_transfer_copy
import logging
import pathlib
import numpy as np

if __name__ == '__main__':
    def output_fcn(self):
        name = 'cable_voltage_driven'
        pathlib.Path('results/' + name).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)


    cable_0 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=129)
    cable_1 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=33)
    cable_2 = cable_voltage_driven.CableVoltageDriven(nonlinear=False, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=5)
    problem = [cable_0, cable_1, cable_2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.Mgrit(problem=problem, transfer=transfer, nested_iteration=True, it=5, debug_lvl=logging.INFO)
    result = mgrit.solve()
