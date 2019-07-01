from mgrit import mgrit_fas as solver
from cable_current_driven import cable_current_driven
from cable_current_driven import grid_transfer_copy

if __name__ == '__main__':
    cable_o = cable_current_driven.CableCurrentDriven(nonlinear=True, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=65)
    cable_1 = cable_current_driven.CableCurrentDriven(nonlinear=True, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=17)
    cable_2 = cable_current_driven.CableCurrentDriven(nonlinear=True, pwm=True, name='cable_2269',
                                                      t_start=0, t_stop=0.02, nt=5)
    problem = [cable_o, cable_1, cable_2]
    transfer = [grid_transfer_copy.GridTransferCopy(), grid_transfer_copy.GridTransferCopy()]
    mgrit = solver.MgritFas(problem=problem, grid_transfer=transfer)
    u = mgrit.solve()
