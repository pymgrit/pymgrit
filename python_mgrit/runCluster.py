from mgrit import mgrit_fas as solver
from mgrit import mgrit_fas_s as solverS
from mgrit import mgrit_fas_point_convergence as solver2
from mgrit import mgrit_fas_machine as solver3
from cable_voltage_driven import cable_voltage_driven
from cable_current_driven import cable_current_driven
from heatEquation import heat_equation
from induction_machine import im_3kW
from scipy import linalg as la
import os

if __name__ == '__main__':

    os.chdir('mgrit')

    #heat = heat_equation.HeatEquation(x_start=0, x_end=2, nx=[1001], t_start=0, t_stop=2, nt=10001)
    #mgrit = solver.MgritFas(problem=heat)
    #mgrit.setup(lvl_max=8, m=2, it=3, tol=1e-7, nested_iteration=True, spatial_coarsening=False, spatial_max_lvl=0)
    #u = mgrit.solve(cf_iter=1, cycle_type='V')
#
    #heat = cable_current_driven.CableCurrentDriven(linear=True, pwm=True, name='modelProblem1', coarse_smooth=False,
    #                                               t_start=0, t_stop=0.02, nt=1001)
    #mgrit = solver.MgritFas(problem=heat)
    #mgrit.setup(lvl_max=3, m=8, it=3, tol=1e-7, nested_iteration=True, spatial_coarsening=False, spatial_max_lvl=0)
    #u = mgrit.solve(cf_iter=1, cycle_type='V')

    heat = cable_voltage_driven.CableVoltageDriven(linear=True, pwm=True, name='modelProblem1', coarse_smooth=False,
                                                   t_start=0, t_stop=0.02, nt=1001)
    mgrit = solver.MgritFas(problem=heat)
    mgrit.setup(lvl_max=5, m=4, it=3, tol=1e-7, nested_iteration=True, spatial_coarsening=False, spatial_max_lvl=0)
    u = mgrit.solve(cf_iter=1, cycle_type='V')


    heat = im_3kW.InductionMachine(nonlinear=False, pwm=True, coarse_smooth=False,
                                   grids=['im_3kW_16k', 'im_3kW_4k'],
                                   additional_unknowns=['jL', 'uA', 'uB', 'uC', 'iA', 'iB', 'iC'], t_start=0,
                                   t_stop=0.0002, nt=65)

    mgrit = solver.MgritFas(problem=heat)
    mgrit2 = solver3.MgritFasMachine(stop_unknown=-7, compute_f_after_convergence=True, problem=heat)

    mgrit.setup(lvl_max=3, m=4, it=3, tol=1e-7, nested_iteration=True, spatial_coarsening=False, spatial_max_lvl=0)
    u = mgrit.solve(cf_iter=1, cycle_type='V')

    mgrit.setup(lvl_max=3, m=4, it=3, tol=1e-7, nested_iteration=True, spatial_coarsening=True, spatial_max_lvl=2)
    u4 = mgrit.solve(cf_iter=1, cycle_type='V')

    mgrit2.setup(lvl_max=3, m=4, it=3, tol=1e-7, nested_iteration=True, spatial_coarsening=False)
    u2 = mgrit2.solve(cf_iter=1, cycle_type='V')

    mgrit2.setup(lvl_max=3, m=4, it=3, tol=1e-7, nested_iteration=True, spatial_coarsening=True, spatial_max_lvl=2)
    u3 = mgrit2.solve(cf_iter=1, cycle_type='V')

