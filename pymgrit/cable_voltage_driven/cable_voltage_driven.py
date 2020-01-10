"""
Problem class for the voltage driven eddy current problem of a coaxial cable
"""

from typing import Dict, Tuple
import numpy as np
from scipy import interpolate
from scipy import linalg as la
from scipy.sparse.linalg import spsolve
from scipy import sparse
import scipy.io as sio

from pymgrit.core import application
from pymgrit.core import vector


class VectorSystem(vector.Vector):
    """
    Solution vector of the system
    """

    def __init__(self, size):
        super(VectorSystem, self).__init__()
        self.size = size
        self.mvp = np.zeros(size)
        self.current = 0

    def __add__(self, other):
        tmp = VectorSystem(self.size)
        tmp.mvp = self.mvp + other.mvp
        tmp.current = self.current + other.current
        return tmp

    def __sub__(self, other):
        tmp = VectorSystem(self.size)
        tmp.mvp = self.mvp - other.mvp
        tmp.current = self.current - other.current
        return tmp

    def norm(self):
        tmp = np.zeros(self.size + 1)
        tmp[:-1] = self.mvp
        tmp[-1] = self.current
        return la.norm(tmp)

    def init_zero(self):
        return VectorSystem(self.size)

    def init_rand(self):
        tmp = VectorSystem(self.size)
        tmp.mvp = np.random.rand(self.size)
        tmp.current = np.random.rand(1)[0]
        return tmp


class CableVoltageDriven(application.Application):
    """
    2-d current driven eddy current problem of a coaxial cable
    sigma dA/dt + Delta x (nu(||Delta x A||) Delta x A) - chi_s*i_s = 0
    d/dt int_sigma(chi_s * A dV) = v_s
    See: https://arxiv.org/abs/1905.06879
    """

    def __init__(self, name, nonlinear, pwm, *args, **kwargs):
        super(CableVoltageDriven, self).__init__(*args, **kwargs)
        self.nonlinear = nonlinear
        path = '/'.join(__file__.split('/')[:-1])
        self.name = name
        self.pwm = pwm
        self.prb = sio.loadmat(path + '/problems/' + name + '.mat', struct_as_record=False, squeeze_me=True)[name]

        intom = 0.0254
        tol = 1e-8
        # r0 = 0.1 * intom
        # r1 = 0.5 * intom
        r2 = 1.0 * intom
        # self.iw = 100
        self.base_frequency = 50  # [Hz] : frequency
        self.pulses = 400  # corresponds to 20000 kHz
        self.omega = 2 * np.pi * self.base_frequency  # [rad/s]: angular frequency
        rg_fe = 3

        bh = np.loadtxt(path + '/problems/BH.txt')

        bchar = bh[:, 0]
        hchar = bh[:, 1]
        self.nlin = self.nlin_initialise(bchar, hchar)

        self.sigmaelem = self.prb_mate_2_elem(self.prb, 'sigma').transpose()[0]
        self.nuelem = self.prb_mate_2_elem(self.prb, 'nu').transpose()
        elemregi = np.zeros_like(self.prb.mesh.elem[:, 3])
        elemregi[self.prb.mesh.elem[:, 3] == 1] = self.prb.region[0, 2]
        elemregi[self.prb.mesh.elem[:, 3] == 2] = self.prb.region[1, 2]
        elemregi[self.prb.mesh.elem[:, 3] == 3] = self.prb.region[2, 2]
        self.idxnlinelem = np.where(elemregi == rg_fe)[0]
        pfem = self.current_pstr(self.prb)
        mfem = self.edgemass_ll(self.prb.mesh, self.sigmaelem)
        kfem = self.curlcurl_ll(self.prb.mesh, self.nuelem)
        x = self.prb.mesh.node[:, 0]
        y = self.prb.mesh.node[:, 1]
        r = self.cart2pol(x, y)
        self.idxdof = np.size(np.where(abs(r - r2) > tol))
        self.points = np.vstack((x, y)).transpose()
        self.msh = mfem[:self.idxdof, :self.idxdof]
        self.ksh = kfem[:self.idxdof, :self.idxdof]
        self.psh = pfem[:self.idxdof, 0].transpose()

        tmp = np.hstack(
            (
                np.vstack((self.msh.toarray(), -self.psh.toarray()[0])),
                np.zeros(self.idxdof + 1)[np.newaxis].T))
        self.a = sparse.csr_matrix(tmp)

        tmp = np.zeros(self.idxdof + 1)
        tmp[:self.psh.shape[1]] = -self.psh.toarray()
        tmp[-1] = -0  # resistance
        tmp = np.hstack((np.vstack((self.ksh.toarray(), np.zeros(self.psh.shape[1]))), tmp[np.newaxis].T))
        self.b = sparse.csr_matrix(tmp)
        self.mesh = self.prb.mesh

        self.u = VectorSystem(self.idxdof)

    def step(self, u_start: VectorSystem, t_start: float, t_stop: float) -> VectorSystem:
        """
        Performing one time step
        :param u_start:
        :param t_start:
        :param t_stop:
        :return:
        """
        tmp = np.zeros(len(u_start.mvp) + 1)
        tmp[:-1] = np.copy(u_start.mvp)
        tmp[-1] = u_start.current
        if self.nonlinear:
            tmp = self.newton(t_start, t_stop, tmp)
        else:
            tmp = self.phi_linear(t_start, t_stop, tmp)
        ret = VectorSystem(u_start.size)
        ret.mvp = tmp[:-1]
        ret.current = tmp[-1]
        return ret

    def phi_linear(self, t_start: float, t_stop: float, vinit: np.ndarray) -> np.ndarray:
        """
        Solves the linear system
        :param t_start:
        :param t_stop:
        :param vinit:
        :return:
        """
        vstepsize = t_stop - t_start

        if self.pwm:
            f = self.a.dot(vinit) / vstepsize + self.rhs_pwm(t_stop, self.a.shape[1])
        else:
            f = self.a.dot(vinit) / vstepsize + self.rhs_sin(t_stop, self.a.shape[1])

        sys = self.a / vstepsize + self.b

        return spsolve(sys, f)

    def rhs_pwm(self, t: float, size: int) -> np.ndarray:
        """
        The pulsed right-hand-side
        :param t:
        :param size:
        :return:
        """
        tmp = np.zeros(size)
        tmp[-1] = 1
        return (1 / 4) * (-self.utl_pwm(t, self.base_frequency, self.pulses) * tmp)

    def rhs_sin(self, t: float, size: int) -> np.ndarray:
        """
        The sine right-hand-side
        :param t:
        :param size:
        :return:
        """
        tmp = np.zeros(size)
        tmp[-1] = 1
        return (1 / 4) * (-np.sin(2 * np.pi * self.base_frequency * t) * tmp)

    @staticmethod
    def utl_pwm(t: float, freq: float, teeth: int) -> float:
        """
        Create pulse width modulation signal
        :param t:
        :param freq:
        :param teeth:
        :return:
        """
        # sawfish pattern with higher frequency
        saw = t * teeth * freq - np.floor(t * teeth * freq)

        # plain sine wave
        sine = np.sin(freq * t * (2 * np.pi))

        # pwm signal by comparison
        pwm = np.sign(sine) * (saw - abs(sine) < 0)

        return pwm

    def newton(self, tstart: float, tstop: float, vinit: np.ndarray) -> np.ndarray:
        """
        Solve the problem using newtons method
        :param tstart:
        :param tstop:
        :param vinit:
        :return:
        """
        max_newton_iterations = 15
        newton_tol = 1e-5
        pwm = self.pwm
        x_old = np.copy(vinit)
        x_new = np.copy(vinit)

        vmass = sparse.hstack(
            (sparse.vstack((self.msh, -self.psh)), np.zeros(self.psh.shape[1] + 1)[np.newaxis].T)) / (tstop - tstart)

        def f(x):
            return vmass.dot(x - x_old) + self.eddy_current_rhs(self.mesh, self.nuelem, self.idxnlinelem,
                                                                self.idxdof, tstop, x, self.psh, self.nlin, pwm)

        def j(x):
            tmp = np.zeros(self.psh.shape[1] + 1)
            tmp[:self.psh.shape[1]] = -self.psh.toarray()
            r = 0
            tmp[-1] = -r
            tmp = tmp[np.newaxis].T
            vjac = self.eddy_current_jac(self.mesh, self.nuelem, self.idxnlinelem, self.idxdof, x, self.nlin)
            vjac = sparse.hstack((sparse.vstack((vjac, np.zeros(self.psh.shape[1]))), tmp))
            return vmass + vjac

        f_value = f(x_new)
        f_norm = la.norm(f_value)  # l2 norm of vector
        iteration_counter = 0
        while abs(f_norm) > newton_tol and iteration_counter < max_newton_iterations:
            delta = spsolve(j(x_new), -f_value)
            x_new = x_new + delta
            f_value = f(x_new)
            f_norm = la.norm(f_value)
            iteration_counter += 1

        return x_new

    def eddy_current_rhs(self, mesh: sio.matlab.mio, nu: np.ndarray, idxnlinelem: np.ndarray, idxdof: int, t: float,
                         ush: np.ndarray, psh: sparse.csc_matrix, nlin: Dict, pwm: bool) -> np.ndarray:
        """
        Constructing the right hand side for eddy current problem
        :param mesh:
        :param nu:
        :param idxnlinelem:
        :param idxdof:
        :param t:
        :param ush:
        :param psh:
        :param nlin:
        :param pwm:
        :return:
        """
        a = np.zeros(np.size(mesh.node, 0))
        a[:idxdof] = ush[:-1]
        b = self.curl(mesh, a)
        bred = b[:, idxnlinelem]
        hred, nured = self.nlin_evaluate(nlin, bred, nargout=2)
        nu[:, idxnlinelem] = np.vstack((nured, nured))
        kfem = self.curlcurl_ll(mesh, nu)

        tmp = np.zeros(psh.shape[1] + 1)
        tmp[:psh.shape[1]] = -psh.toarray()
        tmp[-1] = 0
        tmp = tmp[np.newaxis].T

        if pwm:
            c = self.rhs_pwm(t, psh.shape[1] + 1)
        else:
            c = self.rhs_sin(t, psh.shape[1] + 1)
        b = sparse.hstack((sparse.vstack((kfem[:idxdof, :idxdof], np.zeros(psh.shape[1]))), tmp))
        rhs = b.dot(ush) - c
        return rhs

    def eddy_current_jac(self, mesh: sio.matlab.mio, nu: np.ndarray, idxnlinelem: np.ndarray, idxdof: int,
                         ush: np.ndarray, nlin: Dict) -> sparse.csr_matrix:
        """
        Constructs the Jacobi Matrix
        :param mesh:
        :param nu:
        :param idxnlinelem:
        :param idxdof:
        :param ush:
        :param nlin:
        :return:
        """
        a = np.zeros(np.size(mesh.node, 0))
        a[:idxdof] = ush[:-1]
        b = self.curl(mesh, a)
        bred = b[:, idxnlinelem]
        hred, nured, nudred, dnud_b2red = self.nlin_evaluate(nlin, bred)
        nu[:, idxnlinelem] = np.vstack((nured, nured))
        dnud_b2 = np.zeros(np.size(b, 1))
        dnud_b2[idxnlinelem] = dnud_b2red

        kfem = self.curlcurl_ll_nonlinear(mesh, b, nu, dnud_b2)  # , Hc)

        ksh = kfem[:idxdof, :idxdof]
        return ksh

    @staticmethod
    def curlcurl_ll(mesh: sio.matlab.mio, nu: np.ndarray) -> sparse.csr_matrix:
        """
        returns the curl-curl matrix for the element-wise constant reluctivities reginu
        :param mesh:
        :param nu:
        :return:
        """
        numnode = np.size(mesh.node, 0)
        i = np.repeat(mesh.elem[:, 0:3], 3)
        j = np.tile(mesh.elem[:, 0:3], 3).flatten()
        v = (np.einsum('ij,ik->ijk', (mesh.b.transpose() * nu[1]).transpose(), mesh.b) +
             np.einsum('ij,ik->ijk', (mesh.c.transpose() * nu[0]).transpose(), mesh.c)).flatten() / np.repeat(
            mesh.area * mesh.depth * 4, 9)
        kfem = sparse.csr_matrix((v, (i - 1, j - 1)), shape=(numnode, numnode))
        return kfem

    @staticmethod
    def curlcurl_ll_nonlinear(mesh: sio.matlab.mio, b: np.ndarray, nu: np.ndarray,
                              dnud_b2: np.ndarray) -> sparse.csr_matrix:
        """
        returns the curl-curl matrix and magnetisation vector for the element-wise magnetic properties
        :param mesh:
        :param b:
        :param nu:
        :param dnud_b2:
        :return:
        """
        numnode = np.size(mesh.node, 0)
        i = np.tile(mesh.elem[:, 0:3], 3).flatten()
        j = np.repeat(mesh.elem[:, 0:3], 3)

        shape = np.dstack((mesh.c, -mesh.b))
        nud = np.einsum('ij,ik->ijk', (2 * b * dnud_b2).transpose(), b.transpose())
        temp = np.zeros_like(nud)
        s0, s1, s2 = temp.shape
        temp.reshape(s0, -1)[:, ::s2 + 1] = np.array([nu[0], nu[0]]).transpose()
        nud = nud + temp
        v = np.einsum('wij,wkj->wik', np.einsum('wij,wjk->wik', shape, nud), shape).flatten() / np.repeat(
            mesh.area * mesh.depth * 4, 9)

        kfem = sparse.csr_matrix((v, (i - 1, j - 1)), shape=(numnode, numnode))
        return kfem

    @staticmethod
    def curl(mesh: sio.matlab.mio, az: np.ndarray) -> np.ndarray:
        """
        computes the magnetic flux density for a given distribution of the line-integrated magnetic vector potential
        :param mesh:
        :param az:
        :return:
        """
        u = np.asarray([az[mesh.elem[:, 0] - 1], az[mesh.elem[:, 1] - 1], az[mesh.elem[:, 2] - 1]]).transpose()

        bp = np.asarray([np.sum(np.multiply(mesh.c, u), 1), -np.sum(np.multiply(mesh.b, u), 1)])
        denom = 2 * np.multiply(mesh.area, mesh.depth)
        b = bp / [denom, denom]
        return b

    def nlin_evaluate(self, nlin: Dict, b: np.ndarray, nargout: int = 4) -> Tuple:
        """
        evaluates a nonlinear material characteristic for a given abscis input determining the working point
        :param nlin:
        :param b:
        :param nargout:
        :return:
        """
        # A. Initialisation
        bm, bangle = self.pyth(b)
        idxleft = np.nonzero(bm < nlin['Bmin'])
        idxright = np.nonzero(bm > nlin['Bmax'])

        # B. Determine the ordinate values
        ppval = interpolate.PPoly(nlin['spline'].c, nlin['spline'].x)
        hm = ppval(bm)
        hm[idxleft] = nlin['initialslope'] * bm[idxleft]
        hm[idxright] = nlin['finalcoercitivity'] + nlin['finalslope'] * bm[idxright]
        h = bangle * np.vstack((hm, hm))

        # C. Determine the chord reluctivity (slope between the (0,0) data point and the working point)
        nu = self.save_divide(hm, bm, nlin['initialslope'])

        if nargout == 2:
            return h, nu

        # D. Determine the differential reluctivity (slope of the line tangential to the nonlinear characteristic at the
        # working point)
        ppval = interpolate.PPoly(nlin['splineder'].c, nlin['splineder'].x)
        nud = ppval(bm)
        nud[idxleft] = nlin['initialslope']
        nud[idxright] = nlin['finalslope']

        dnud_b2 = self.save_divide(nud - nu, 2 * bm ** 2)

        return h, nu, nud, dnud_b2

    @staticmethod
    def pyth(b: np.ndarray) -> Tuple:
        """
        returns the magnitude of the vector/coordinate B
        :param b:
        :return:
        """
        bm = np.sqrt(np.sum(np.multiply(b, np.conj(b)), 0))
        # if nargout == 2:
        bangle = b
        idx = np.nonzero(bm)[0]
        bangle[:, idx] = bangle[:, idx] / np.vstack((bm[idx], bm[idx]))
        return bm, bangle
        # return bm

    @staticmethod
    def save_divide(x: np.ndarray, y: np.ndarray, value: np.ndarray = 0) -> np.ndarray:
        """
        divides vector x through vector y avoiding division through zero and introducing value instead
        :param x:
        :param y:
        :param value:
        :return:
        """
        z = value * np.ones(np.size(x))
        i = y.ravel().nonzero()
        z[i] = x[i] / y[i]
        return z

    @staticmethod
    def nlin_initialise(bchar: np.ndarray, hchar: np.ndarray) -> Dict:
        """
        initialises the data for further use when evaluating nonlinear material characteristics
        :param bchar:
        :param hchar:
        :return:
        """
        nlin = {'B': np.sort(np.abs(bchar)), 'H': np.sort(np.abs(hchar))}

        if nlin['B'][0] == 0:
            nlin['B'] = nlin['B'][1:]
            nlin['H'] = nlin['H'][1:]

        nlin['Bmin'] = np.min(nlin['B'])
        nlin['Bmax'] = np.max(nlin['B'])
        nlin['Hmin'] = np.min(nlin['H'])
        nlin['Hmax'] = np.max(nlin['H'])

        nlin['initialslope'] = np.array([nlin['H'][0] / nlin['B'][0]])

        nlin['finalslope'] = np.diff(nlin['H'][-2:]) / np.diff(nlin['B'][-2:])

        nlin['finalcoercitivity'] = nlin['H'][-1] - nlin['B'][-1] * nlin['finalslope']
        nlin['finalremanence'] = nlin['B'][-1] - nlin['H'][-1] / nlin['finalslope']

        nlin['spline'] = interpolate.CubicSpline(nlin['B'], nlin['H'],
                                                 bc_type=((1, nlin['initialslope'][0]), (1, nlin['finalslope'][0])))
        nlin['splineder'] = nlin['spline'].derivative()
        nlin['splineint'] = nlin['spline'].antiderivative()
        ppval = interpolate.PPoly(nlin['splineint'].c, nlin['splineint'].x)
        nlin['finalWmagn'] = ppval(nlin['B'][-1])

        return nlin

    @staticmethod
    def cart2pol(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Transform Cartesian coordinates to polar or cylindrical
        :param x:
        :param y:
        :return:
        """
        r = (x ** 2 + y ** 2) ** .5
        return r

    @staticmethod
    def current_pstr(prb: sio.matlab.mio) -> sparse.csr_matrix:
        """
        computes the coupling blocks for stranded conductors
        :param prb:
        :return:
        """
        numnode = np.size(prb.mesh.node, 0)
        numelem = np.size(prb.mesh.elem, 0)
        numregi = np.size(prb.region, 0)
        numwire = np.size(prb.wire)

        regionarea = np.zeros(numregi)
        fillfactor = np.zeros(numregi)
        for rg in range(0, numregi):
            regionarea[rg] = np.sum(prb.mesh.area[np.where(prb.mesh.elem[:, 3] == (rg + 1))])
            mt = int(prb.region[rg, 2])
            nt = int(prb.region[rg, 7])
            if prb.material[mt - 1].wireD != 0:
                fillfactor[rg - 1] = nt * np.pi * (prb.material[mt].wireD / 2) ** 2 / regionarea[rg]
            else:
                fillfactor[rg - 1] = 1

        pstr = np.zeros((numnode, numwire))

        for k in range(numelem):
            rg = prb.mesh.elem[k, 3] - 1
            wr = int(prb.region[rg, 4])
            nt = prb.region[rg, 7]
            if wr != 0:
                idx = prb.mesh.elem[k, 0:3]
                pstr[idx - 1, wr - 1] = pstr[idx - 1, wr - 1] + np.array(
                    [np.ones(3) * nt / regionarea[rg] * prb.mesh.area[k] / 3])

        idxi = np.where(prb.mesh.node[:, 3] != 0)[0]
        for iii in range(np.size(idxi)):
            i = idxi[iii]
            pt = int(prb.mesh.node[i, 3])
            wr = prb.geometry.points[pt - 1, 4]
            if wr != 0:
                pstr[i, wr] = pstr[i, wr] + 1

        return sparse.csr_matrix(pstr)

    @staticmethod
    def edgemass_ll(mesh: sio.matlab.mio, sigma: np.ndarray) -> sparse.csr_matrix:
        """
        returns the cff-edge-mass matrix for a certain element-wise constant conductivity
        :param mesh:
        :param sigma:
        :return:
        """
        numnode = np.size(mesh.node, 0)
        numelem = np.size(mesh.elem, 0)
        i = np.repeat(mesh.elem[:, 0:3], 3)
        j = np.tile(mesh.elem[:, 0:3], 3).flatten()
        v = np.repeat(mesh.area, 9) / 12 * np.tile(np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]).flatten(),
                                                   numelem) * np.repeat(sigma, 9) / np.repeat(mesh.depth, 9)

        mfem = sparse.csr_matrix((v, (i - 1, j - 1)), shape=(numnode, numnode))
        return mfem

    @staticmethod
    def prb_mate_2_elem(prb: sio.matlab.mio, mateprop: str) -> np.ndarray:
        """
        translates a material-wise property into an element-wise property
        :param prb:
        :param mateprop:
        :return:
        """
        prop = mateprop

        if prop == 'nu':
            mateprop = np.array(
                [[795774.71545948, 795774.71545948], [795774.71545948, 795774.71545948], [795.77471546, 795.77471546]])
        elif prop == 'sigma':
            mateprop = np.array(
                [[0], [0], [10000000]])
        else:
            raise Exception("error")

        rg = prb.mesh.elem[:, 3]
        mt = np.array(prb.region[rg - 1, 2], dtype=int)
        elemprop = mateprop[mt - 1, :]

        return elemprop
