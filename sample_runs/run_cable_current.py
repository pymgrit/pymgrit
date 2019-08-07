from mgrit import mgrit_fas as solver
from cable_current_driven import cable_current_driven
from cable_current_driven import grid_transfer_copy
import pathlib
import numpy as np

if __name__ == '__main__':
    def output_fcn(self):
        name = 'run_cable_current'
        pathlib.Path('results/' + name).mkdir(parents=True, exist_ok=True)
        sol = {'u': [self.u[0][i] for i in self.index_local[0]], 'time': self.runtime_solve, 'conv': self.conv,
               't': self.problem[0].t, 'time_setup': self.runtime_setup}

        np.save('results/' + name + '/' + str(self.t[0][0]) + '-' + str(self.t[0][-1]), sol)

    cable_0 = cable_current_driven.CableCurrentDriven(nonlinear=True, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=65)
    cable_1 = cable_current_driven.CableCurrentDriven(nonlinear=True, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=17)
    cable_2 = cable_current_driven.CableCurrentDriven(nonlinear=True, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=5)
    problem = [cable_0, cable_1, cable_2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, transfer=transfer)
    u = mgrit.solve()
