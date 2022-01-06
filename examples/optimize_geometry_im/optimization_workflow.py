"""
Parallel version of the workflow

Installation guide:
    pip3 install Py-BOBYQA
    pip3 install pymgrit

Required:
    Add path to the model in main
    Add path to getdp in main

Run (at least two processors needed: Master and Worker):
    mpirun -np 2 python3 workflow_cluster.py
"""

import pybobyqa
import time
import shutil
import numpy as np
import subprocess
from subprocess import PIPE

from mpi4py import MPI

from pymgrit.core.at_mgrit import AtMgrit
from pymgrit.induction_machine.induction_machine import InductionMachine


class AtMgritCustomized(AtMgrit):
    def __init__(self, objective_function, region_from_end, *args, **kwargs) -> None:
        self.optimization_region = region_from_end
        self.objective_function = objective_function
        super().__init__(*args, **kwargs)
        self.last_it = np.zeros_like(self.problem[0].t)
        self.convergence_criterion(0)

    # Find time point closest to desired time point
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def solve(self):
        super().solve()
        vals = [[self.u[0][i].tr, self.u[0][i].jl] for i in self.index_local[0]]
        tmp = self.comm_time.allgather(vals)
        flat_list = [item for sublist in tmp for item in sublist]
        tmp = np.array(flat_list)
        closest_value, closest_idx = self.find_nearest(self.problem[0].t,
                                                       self.problem[0].t[-1] - self.optimization_region)
        tr = np.mean(tmp[closest_idx:, 0])
        jl = np.mean(tmp[closest_idx:, 1])
        return tmp[:, 0], tmp[:, 1], tr, jl

    def convergence_criterion(self, iteration: int) -> None:
        vals = [[self.u[0][i].tr, self.u[0][i].jl] for i in self.index_local[0]]
        tmp = self.comm_time.allgather(vals)
        flat_list = [item for sublist in tmp for item in sublist]
        tmp = np.array(flat_list)
        tr = tmp[:, 0]
        jl = tmp[:, 1]
        closest_value, closest_idx = self.find_nearest(self.problem[0].t,
                                                       self.problem[0].t[-1] - self.optimization_region)

        tmp = 100 * np.max(
            np.abs(np.abs(np.divide((jl[closest_idx:] - self.last_it[closest_idx:]), jl[closest_idx:],
                                    out=np.zeros_like(self.last_it[closest_idx:]), where=jl[closest_idx:] != 0))))

        self.log_info(
            f"{iteration}  'closest_idx:' {closest_idx} 'tmp:' {tmp} 'closest_idx:' {closest_idx}")
        self.conv[iteration] = tmp
        self.last_it = np.copy(jl)

    def nested_iteration(self) -> None:
        """
        Generates an initial approximation on the finest grid
        by solving the problem on the coarsest grid and interpolating
        the approximation to the finest level.
        """
        if self.comm_time_rank == 0:
            self.u_coarsest = [self.problem[-1].vector_t_start.clone()]
            for i in range(1, len(self.global_t[-1])):
                self.u_coarsest.append(self.step[-1](u_start=self.u_coarsest[i - 1],
                                                     t_start=self.global_t[-1][i - 1],
                                                     t_stop=self.global_t[-1][i]))
                if self.comm_coarsest_level[i] != self.comm_time_rank:
                    req_s = self.comm_time.send([self.global_t[-1][i], self.u_coarsest[i].pack()],
                                                dest=self.comm_coarsest_level[i],
                                                tag=self.comm_coarsest_level[i])
                else:
                    idx = np.where(self.global_t[-1][i] == self.t[-1])[0][0]
                    self.u[-1][idx] = self.u_coarsest[i]

        else:
            for i in range(self.cpts[-1].size):
                recv = self.comm_time.recv(source=0, tag=self.comm_time_rank)
                idx = np.where(recv[0] == self.t[-1])[0][0]
                self.u[-1][idx].unpack(recv[1])

        for lvl in range(self.lvl_max - 2, -1, -1):
            for i in range(len(self.index_local[lvl + 1])):
                self.u[lvl][self.index_local_c[lvl][i]] = self.interpolation[lvl](
                    u=self.u[lvl + 1][self.index_local[lvl + 1][i]])

            self.f_exchange(lvl)
            self.c_exchange(lvl)
            if lvl > 0:
                self.iteration(lvl=lvl, cycle_type='V', iteration=0, first_f=True)


def run_mgrit(exe_path, model_path, t_stop, nt, comm):
    path_im3kw = model_path
    path_getdp = exe_path + 'getdp'

    machine_0 = InductionMachine(nonlinear=True, pwm=False, grid='im_3kW', t_start=0, t_stop=t_stop, nt=nt,
                                 path_getdp=path_getdp, path_im3kw=path_im3kw, imposed_speed=2,
                                 stop_criterion=1e-06)
    machine_1 = InductionMachine(nonlinear=True, pwm=False, grid='im_3kW', t_interval=machine_0.t[::64],
                                 path_getdp=path_getdp, path_im3kw=path_im3kw, imposed_speed=2,
                                 stop_criterion=1e-06)

    problem = [machine_0, machine_1]
    mgrit = AtMgritCustomized(problem=problem, nested_iteration=True, comm_time=comm, tol=1,
                              objective_function=objective_function, region_from_end=0.02, cf_iter=0, k=100)
    return mgrit.solve()


def create_mesh(exe_path, model_path, Rsl=0.00213, h2=0.01425):
    exe_string = [
        exe_path + 'gmsh',
        model_path + 'im_3kW.geo',
        '-2',
        '-setnumber Rsl', str(Rsl),
        '-setnumber h2', str(h2),
        '-o', model_path + 'im_3kW.msh'
    ]

    status = subprocess.run(' '.join(exe_string), shell=True, stdout=PIPE)

    # create Pre file
    exe_string = [
        exe_path + 'getdp',
        model_path + 'im_3kW.pro',
        '-pre "#1"',
        '-msh', model_path + 'im_3kW.msh',
        '-name', model_path + 'im_3kW',
        '-res', model_path + 'im_3kW.res',
        '-setstring ResDir', model_path + 'res/',
        '-setnumber Flag_AnalysisType 1 -setnumber Flag_NL 0 -setnumber Flag_ImposedSpeed 2 -setnumber Nb_max_iter 60 -setnumber relaxation_factor 0.5 -setnumber stop_criterion 1e-06 -setnumber NbTrelax 2 -setnumber Flag_PWM 0'
    ]

    status = subprocess.run(' '.join(exe_string), shell=True, stdout=PIPE)


def objective_function(tr, jl):
    return -((tr * 148.7) / ((tr * 148.7) + jl))


def objx(x, exe_path, model_path):
    # Print current rotor setting
    print(x)

    # createMesh
    create_mesh(Rsl=x[0], h2=x[1],
                exe_path=exe_path,
                model_path=model_path)

    # Copy mesh (optional)
    # copy_path = '' #Add path if mesh is to be copied
    # shutil.copy(model_path + 'im_3kW.msh',
    #             copy_path + 'im_3kW_' + str(objx.counter) + '_' + str(x[0]) + '_' + str(x[1]) + '.msh')

    objx.counter += 1

    # Wait briefly to make created mesh available
    time.sleep(10)

    # Not finished yet
    MPI.COMM_WORLD.bcast(0, root=0)

    # Receive tr and jl
    obj, obj2 = MPI.COMM_WORLD.bcast([1, 1], root=MPI.COMM_WORLD.Get_size() - 1)

    ret_obj = objective_function(tr=obj, jl=obj2)
    print(ret_obj)

    return ret_obj


def worker(comm, t_stop, nt, model_path, exe_path):
    finished = 0
    counter = 0
    while True:
        finished = MPI.COMM_WORLD.bcast(finished, root=0)
        if finished:
            break
        else:
            tr_arr, jl_arr, tr, jl = run_mgrit(model_path=model_path,
                                               exe_path=exe_path,
                                               t_stop=t_stop, nt=nt,
                                               comm=comm)
        # Optional save results
        # save_path = ''  # Add path to save run results
        # if comm.Get_rank() == 0:
        #    np.savez(save_path + str(counter), tr=tr_arr, jl=jl_arr)

        counter = counter + 1
        MPI.COMM_WORLD.bcast([tr, jl], root=MPI.COMM_WORLD.Get_size() - 1)


if __name__ == '__main__':

    # Setup communication
    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    if rank == 0:
        color = 1
    else:
        color = 0

    worker_comm = comm_world.Split(color, key=rank)

    # attribute must be initialized
    objx.counter = 0

    # Setup optimization
    # width Rsl, height h2
    x0 = np.array([0.002, 0.01425])
    lower = np.array([0.0015, 0.007])
    upper = np.array([0.0035, 0.015])

    # Setup interval
    t_stop = 0.2
    nt = 2 ** 14 + 1

    # Define getdp and model path
    exe_path = ''  # Path to getDP
    model_path = ''  # Path to the induction machine model

    # Run
    if rank == 0:
        soln = pybobyqa.solve(objx, x0, args=(exe_path, model_path), bounds=(lower, upper), rhobeg=.0001,
                              rhoend=.000001)
        MPI.COMM_WORLD.bcast(1, root=0)
        print(soln)
    else:
        worker(comm=worker_comm, t_stop=t_stop, nt=nt, model_path=model_path, exe_path=exe_path)
