"""
Application class for the model 'im_3_kW' of an induction machine
https://gitlab.onelab.info/doc/models/-/wikis/Electric-machines
"""

from typing import Dict
import os
import subprocess
from subprocess import PIPE
import tempfile
import time

import numpy as np

from pymgrit.core.application import Application
from pymgrit.induction_machine.vector_machine import VectorMachine
from pymgrit.induction_machine.helper import is_numeric, pre_file, get_values_from, getdp_read_resolution, \
    set_resolution, get_preresolution


class InductionMachine(Application):
    """
    Simulating an induction machine using the model 'im_3_kW' from
    https://gitlab.onelab.info/doc/models/-/wikis/Electric-machines
    """

    def __init__(self, grid: str, path_im3kw: str, path_getdp: str, imposed_speed: int = 1, nb_trelax: int = 2,
                 analysis_type: int = 1, nb_max_iter: int = 60, relaxation_factor: float = 0.5,
                 stop_criterion: float = 1e-6, nonlinear: bool = False, pwm: bool = False, pro_file: str = 'im_3kW.pro',
                 verbose: bool = False, *args, **kwargs):
        """
        Constructor

        :param nonlinear: Nonlinear or linear model
        :param pwm: pwm or sin rhs
        :param grid: mesh
        :param path_im3kw: path to im_3kW data
        :param path_getdp: path to getdp
        :param imposed_speed: imposed speed
        """
        super().__init__(*args, **kwargs)

        self.pro_path = path_im3kw + pro_file
        if not os.path.isfile(self.pro_path):
            raise Exception('Found no valid .pro file in', self.pro_path)

        self.getdp_path = path_getdp
        if not os.path.isfile(self.getdp_path):
            raise Exception('Getdp not found (http://getdp.info/)')

        self.nl = int(nonlinear)
        self.pwm = int(pwm)
        self.mesh = grid + '.msh'
        self.pre = grid + '.pre'
        self.further_unknowns_front = 8
        self.further_unknowns_back = 15

        cor_to_un, un_to_cor, boundary = pre_file(path_im3kw + self.pre)

        self.nx = len(un_to_cor) + self.further_unknowns_front + self.further_unknowns_back

        self.gopt = {'Verbose': int(verbose), 'TimeStep': self.t[1] - self.t[0], 'Executable': self.getdp_path,
                     'PreProcessing': '#1'}
        self.fopt = ['Flag_AnalysisType', analysis_type, 'Flag_NL', self.nl, 'Flag_ImposedSpeed', imposed_speed,
                     'Nb_max_iter', nb_max_iter, 'relaxation_factor', relaxation_factor, 'stop_criterion',
                     stop_criterion, 'NbTrelax', nb_trelax, 'Flag_PWM', self.pwm]

        version_test = subprocess.run([self.gopt['Executable'], '--version'], stdout=PIPE, stderr=PIPE)
        if version_test.returncode:
            raise Exception('getdp not found.')

        self.vector_template = VectorMachine(u_front_size=self.further_unknowns_front,
                                             u_back_size=self.further_unknowns_back,
                                             u_middle_size=len(un_to_cor))
        self.vector_t_start = VectorMachine(u_front_size=self.further_unknowns_front,
                                            u_back_size=self.further_unknowns_back,
                                            u_middle_size=len(un_to_cor))

    def step(self, u_start: VectorMachine, t_start: float, t_stop: float) -> VectorMachine:
        """
        Time integration routine

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        soli = self.run_getdp(u_start=u_start.get_values(), t_start=t_start, t_stop=t_stop)
        ret = VectorMachine(u_front_size=u_start.u_front_size,
                            u_back_size=u_start.u_back_size, u_middle_size=u_start.u_middle_size)

        ret.set_values(values=soli['y'][-1], jl=soli['jl'][-1], ia=soli['ia'][-1], ib=soli['ib'][-1], ic=soli['ic'][-1],
                       ua=soli['ua'][-1], ub=soli['ub'][-1], uc=soli['uc'][-1], tr=soli['tr'][-1])
        return ret

    def run_getdp(self, u_start: np.ndarray, t_start: float, t_stop: float) -> Dict:
        """
        Calls getdp

        :param u_start: approximate solution for the input time t_start
        :param t_start: time associated with the input approximate solution u_start
        :param t_stop: time to evolve the input approximate solution to
        :return: approximate solution at input time t_stop
        """
        if np.max(np.isnan(u_start)):
            raise Exception('Approximation contains nan')

        fdir, file = os.path.split(self.pro_path)
        fname, fext = os.path.splitext(file)

        funargstr = ''
        for i in range(0, len(self.fopt), 2):
            if is_numeric(self.fopt[i + 1]):
                funargstr = funargstr + ' -setnumber ' + str(self.fopt[i]) + ' ' + str(self.fopt[i + 1])
            else:
                funargstr = funargstr + ' -setstring ' + str(self.fopt[i]) + ' ' + self.fopt[i + 1]
        funargstr = funargstr[1:]

        mshfile = os.path.join(fdir, self.mesh)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_name = os.path.join(tmpdir, fname)
            resdir = os.path.join(tmpdir, 'res')
            prefile = os.path.join(tmpdir, fname + '.pre')
            resfile = os.path.join(tmpdir, fname + '.res')
            joule_file = os.path.join(tmpdir, 'resJL.dat')
            ua_file = os.path.join(tmpdir, 'resUa.dat')
            ub_file = os.path.join(tmpdir, 'resUb.dat')
            uc_file = os.path.join(tmpdir, 'resUc.dat')
            ia_file = os.path.join(tmpdir, 'resIa.dat')
            ib_file = os.path.join(tmpdir, 'resIb.dat')
            ic_file = os.path.join(tmpdir, 'resIc.dat')
            tr_file = os.path.join(tmpdir, 'resTr.dat')

            # Preprocessing
            exe_string = [self.gopt['Executable'],
                          self.pro_path,
                          '-pre "' + self.gopt['PreProcessing'] + '"',
                          '-msh', mshfile,
                          '-name', tmp_name,
                          '-res', resfile,
                          '-setnumber timemax', str(t_stop),
                          '-setnumber dtime', str(self.gopt['TimeStep']),
                          '-setstring ResDir', resdir,
                          funargstr]

            if self.gopt['Verbose'] == 1:
                status = subprocess.run(' '.join(exe_string), shell=True)
            else:
                status = subprocess.run(' '.join(exe_string), shell=True, stdout=PIPE, stderr=PIPE)

            if status.returncode:
                raise Exception('preprocessing failed')

            num_dofs = np.size(u_start)
            num_pres = get_preresolution(file=prefile)

            if num_dofs != np.sum(num_pres):
                raise Exception(
                    'u_start has wrong size: ' + str(num_dofs) + ' instead of ' + str(num_pres) + ': ' + str(prefile))

            # Create initial data
            set_resolution(file=resfile, t_start=t_start, u_start=u_start, num_dofs=num_dofs)

            # Compute solution
            exe_string = [self.gopt['Executable'],
                          self.pro_path,
                          '-restart',
                          '-msh', mshfile,
                          '-name', tmp_name,
                          '-res', resfile,
                          '-setnumber timemax', str(t_stop),
                          '-setnumber dtime', str(self.gopt['TimeStep']),
                          '-setstring ResDir', resdir,
                          funargstr]

            if self.gopt['Verbose'] == 1:
                status = subprocess.run(' '.join(exe_string), shell=True)
            else:
                status = subprocess.run(' '.join(exe_string), shell=True, stdout=PIPE, stderr=PIPE)
            if status.returncode:
                raise Exception('getdp solving failed')

            # Read results
            t, y = getdp_read_resolution(file=resfile, num_dofs=num_dofs)
            jl = get_values_from(file=joule_file)
            ia = get_values_from(file=ia_file)
            ib = get_values_from(file=ib_file)
            ic = get_values_from(file=ic_file)
            ua = get_values_from(file=ua_file)
            ub = get_values_from(file=ub_file)
            uc = get_values_from(file=uc_file)
            tr = get_values_from(file=tr_file)

        return {'x': t, 'y': y, 'jl': jl, 'ia': ia, 'ib': ib, 'ic': ic, 'ua': ua, 'ub': ub, 'uc': uc, 'tr': tr}
