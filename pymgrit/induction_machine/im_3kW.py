from typing import Tuple, List, Dict
import time
import numpy as np

from pymgrit.core import application
from pymgrit.induction_machine import odegetdp
from pymgrit.induction_machine import vector_machine


class InductionMachine(application.Application):
    """
    Simulating an induction machine using the model... TODO
    """

    def __init__(self, nonlinear, pwm, grid, *args, **kwargs):
        super(InductionMachine, self).__init__(*args, **kwargs)

        path = '/'.join(__file__.split('/')[:-1])
        self.pro_path = path + '/im_3kW/im_3kW.pro'
        self.getdp_path = path + '/getdp/getdp'

        self.nl = int(nonlinear)
        self.pwm = int(pwm)
        self.odegetdp = odegetdp.odegetdp
        self.mesh = grid + '.msh'
        self.pre = grid + '.pre'
        self.further_unknowns_front = 8
        self.further_unknowns_back = 15

        cor_to_un, un_to_cor, boundary = self.pre_file(path + '/im_3kW/' + self.pre)

        self.nx = len(un_to_cor) + self.further_unknowns_front + self.further_unknowns_back

        self.gopt = {'Verbose': 0, 'TimeStep': self.t[1] - self.t[0], 'Executable': self.getdp_path}
        self.fopt = ['Flag_AnalysisType', 1, 'Flag_NL', self.nl, 'Flag_ImposedSpeed', 1, 'Nb_max_iter', 60,
                     'relaxation_factor', 0.5, 'stop_criterion', 1e-6, 'NbTrelax', 2, 'Flag_PWM', self.pwm]

        self.u = vector_machine.VectorMachine(u_front_size=self.further_unknowns_front,
                                              u_back_size=self.further_unknowns_back,
                                              u_middle_size=len(un_to_cor))

        # self.count_solves = 0
        # self.time_solves = 0

    def step(self, u_start: vector_machine.VectorMachine, t_start: float,
             t_stop: float) -> vector_machine.VectorMachine:
        """
        Perform one time step
        :param u_start:
        :param t_start:
        :param t_stop:
        :return:
        """
        # start = time.time()
        tmp = np.append(u_start.u_front, u_start.u_middle)
        tmp = np.append(tmp, u_start.u_back)

        soli = self.odegetdp(self.pro_path, np.array([t_start, t_stop]), tmp,
                             self.gopt, self.fopt, self.mesh)
        ret = vector_machine.VectorMachine(u_front_size=u_start.u_front_size,
                                           u_back_size=u_start.u_back_size, u_middle_size=u_start.u_middle_size)

        ret.u_front = soli['y'][-1][:u_start.u_front_size]
        ret.u_middle = soli['y'][-1][u_start.u_front_size:-u_start.u_back_size]
        ret.u_back = soli['y'][-1][-u_start.u_back_size:]
        ret.jl = soli['jl'][-1]
        ret.ia = soli['ia'][-1]
        ret.ib = soli['ib'][-1]
        ret.ic = soli['ic'][-1]
        ret.ua = soli['ua'][-1]
        ret.ub = soli['ub'][-1]
        ret.uc = soli['uc'][-1]
        # self.time_solves += time.time() - start
        # self.count_solves += 1
        return ret

    @staticmethod
    def pre_file(file: str) -> Tuple[Dict, Dict, List]:
        """
        Read prefile and return mapping between nodes
        :param file:
        :return:
        """
        with open(file) as f:
            content = f.readlines()

        mapping = content[9:-35]

        cor_to_un = {}
        un_to_cor = {}
        boundary = []

        for ma in mapping:
            row = ma.split()
            if row[4] != '0' and row[4] != '-1' and row[4] != '1':
                cor_to_un[row[1]] = row[4]
                un_to_cor[row[4]] = row[1]
            else:
                boundary = boundary + [row[1]]
        return cor_to_un, un_to_cor, boundary
