from pymgrit.core import application
from pymgrit.cable_current_driven import vector_standard
import scipy.sparse as sp
from scipy import interpolate
from scipy import linalg as la
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.io as sio


class CableCurrentDriven(application.Application):
    """
    """

    def __init__(self, name, nonlinear, pwm, *args, **kwargs):
        super(CableCurrentDriven, self).__init__(*args, **kwargs)
        self.nonlinear = nonlinear
        path = '/'.join(__file__.split('/')[:-1])
        self.name = name
        self.pwm = pwm
        self.prb = sio.loadmat(path + '/problems/' + name + '.mat', struct_as_record=False, squeeze_me=True)[name]
        self.nx = 0

        intom = 0.0254
        tol = 1e-8
        # r0 = 0.1 * intom
        # r1 = 0.5 * intom
        r2 = 1.0 * intom
        self.iw = 100
        self.base_frequency = 50
        self.pulses = 400  # corresponds to 20000 kHz
        self.omega = 2 * np.pi * self.base_frequency
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
        self.mesh = self.prb.mesh

        self.u = vector_standard.VectorStandard(self.idxdof)

    def step(self, u_start: vector_standard.VectorStandard, t_start: float,
             t_stop: float) -> vector_standard.VectorStandard:
        tmp = np.copy(u_start.vec)
        if self.nonlinear:
            tmp = self.newton(t_start, t_stop, tmp)
        else:
            tmp = self.phi_linear(t_start, t_stop, tmp)
        ret = vector_standard.VectorStandard(u_start.size)
        ret.vec = tmp
        return ret

    def phi_linear(self, t_start, t_stop, vinit):
        vstepsize = t_stop - t_start

        if self.pwm:
            f = self.msh.dot(vinit) / vstepsize + \
                (self.psh * self.iw * self.utl_pwm(t_stop, self.base_frequency, self.pulses)).toarray()[0]
        else:
            f = self.msh.dot(vinit) / vstepsize + \
                (self.psh * self.iw * np.sin(self.omega * t_stop)).toarray()[0]

        return spsolve(self.msh / vstepsize + self.ksh, f)

    @staticmethod
    def utl_pwm(t, freq, teeth):
        # sawfish pattern with higher frequency
        saw = t * teeth * freq - np.floor(t * teeth * freq)

        # plain sine wave
        sine = np.sin(freq * t * (2 * np.pi))

        # pwm signal by comparison
        pwm = np.sign(sine) * (saw - abs(sine) < 0)

        return pwm

    def newton(self, tstart, tstop, vinit):
        max_newton_iterations = 15
        newton_tol = 1e-5
        pwm = self.pwm
        xold = np.copy(vinit)
        xnew = np.copy(vinit)

        vmass = self.msh / (tstop - tstart)

        def f(x):
            return vmass.dot(x - xold) - self.eddy_current_rhs(self.mesh, self.nuelem, self.idxnlinelem,
                                                               self.idxdof, tstop, x, self.psh, self.iw,
                                                               self.omega, self.nlin, pwm)

        def j(x):
            return vmass - self.eddy_current_jac(self.mesh, self.nuelem, self.idxnlinelem, self.idxdof, x, self.nlin)

        f_value = f(xnew)
        f_norm = la.norm(f_value, np.inf)  # l2 norm of vector
        iteration_counter = 0
        while abs(f_norm) > newton_tol and iteration_counter < max_newton_iterations:
            delta = spsolve(j(xnew), -f_value)
            xnew = xnew + delta
            f_value = f(xnew)
            f_norm = la.norm(f_value, np.inf)
            iteration_counter += 1

        return xnew

    def eddy_current_rhs(self, mesh, nu, idxnlinelem, idxdof, t, ush, psh, iw, omega, nlin, pwm):

        a = np.zeros(np.size(mesh.node, 0))
        a[:idxdof] = ush
        b = self.curl(mesh, a)
        bred = b[:, idxnlinelem]
        hred, nured = self.nlin_evaluate(nlin, bred, nargout=2)
        nu[:, idxnlinelem] = np.vstack((nured, nured))
        kfem = self.curlcurl_ll(mesh, nu)

        if pwm:
            rhs = -kfem[:idxdof, :idxdof].dot(ush) + \
                  (psh * iw * self.utl_pwm(t, self.base_frequency, self.pulses)).toarray()[0]
        else:
            rhs = -kfem[:idxdof, :idxdof].dot(ush) + \
                  (psh * iw * np.sin(omega * t)).toarray()[0]
        return rhs

    def eddy_current_jac(self, mesh, nu, idxnlinelem, idxdof, ush, nlin):
        a = np.zeros(np.size(mesh.node, 0))
        a[:idxdof] = ush
        b = self.curl(mesh, a)
        bred = b[:, idxnlinelem]
        hred, nured, nudred, dnud_b2red = self.nlin_evaluate(nlin, bred)
        nu[:, idxnlinelem] = np.vstack((nured, nured))
        dnud_b2 = np.zeros(np.size(b, 1))
        dnud_b2[idxnlinelem] = dnud_b2red

        kfem = self.curlcurl_ll_nonlinear(mesh, b, nu, dnud_b2)  # , Hc)

        ksh = -kfem[:idxdof, :idxdof]
        return ksh

    @staticmethod
    def curlcurl_ll(mesh, nu):
        numnode = np.size(mesh.node, 0)
        i = np.repeat(mesh.elem[:, 0:3], 3)
        j = np.tile(mesh.elem[:, 0:3], 3).flatten()
        v = (np.einsum('ij,ik->ijk', (mesh.b.transpose() * nu[1]).transpose(), mesh.b) + np.einsum('ij,ik->ijk', (
                mesh.c.transpose() * nu[0]).transpose(), mesh.c)).flatten() / np.repeat(mesh.area * mesh.depth * 4, 9)
        kfem = sp.csr_matrix((v, (i - 1, j - 1)), shape=(numnode, numnode))
        return kfem

    @staticmethod
    def curlcurl_ll_nonlinear(mesh, b, nu, dnud_b2):
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

        kfem = sp.csr_matrix((v, (i - 1, j - 1)), shape=(numnode, numnode))
        return kfem

    @staticmethod
    def curl(mesh, az):
        u = np.asarray([az[mesh.elem[:, 0] - 1], az[mesh.elem[:, 1] - 1], az[mesh.elem[:, 2] - 1]]).transpose()

        bp = np.asarray([np.sum(np.multiply(mesh.c, u), 1), -np.sum(np.multiply(mesh.b, u), 1)])
        denom = 2 * np.multiply(mesh.area, mesh.depth)
        b = bp / [denom, denom]
        return b

    def nlin_evaluate(self, nlin, b, nargout=4):
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
        nu = self.savedivide(hm, bm, nlin['initialslope'])

        if nargout == 2:
            return h, nu

        # D. Determine the differential reluctivity (slope of the line tangential to the nonlinear characteristic at the
        # working point)
        ppval = interpolate.PPoly(nlin['splineder'].c, nlin['splineder'].x)
        nud = ppval(bm)
        nud[idxleft] = nlin['initialslope']
        nud[idxright] = nlin['finalslope']

        dnud_b2 = self.savedivide(nud - nu, 2 * bm ** 2)

        return h, nu, nud, dnud_b2

    @staticmethod
    def pyth(b, nargout=2):
        bm = np.sqrt(np.sum(np.multiply(b, np.conj(b)), 0))
        if nargout == 2:
            bangle = b
            idx = np.nonzero(bm)[0]
            bangle[:, idx] = bangle[:, idx] / np.vstack((bm[idx], bm[idx]))
            return bm, bangle
        return bm

    @staticmethod
    def savedivide(x, y, value=0):
        z = value * np.ones(np.size(x))
        i = y.ravel().nonzero()
        z[i] = x[i] / y[i]
        return z

    @staticmethod
    def nlin_initialise(bchar, hchar):
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
    def cart2pol(x, y):
        r = (x ** 2 + y ** 2) ** .5
        return r

    @staticmethod
    def current_pstr(prb):
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

        return sp.csr_matrix(pstr)

    @staticmethod
    def edgemass_ll(mesh, sigma):
        numnode = np.size(mesh.node, 0)
        numelem = np.size(mesh.elem, 0)
        i = np.repeat(mesh.elem[:, 0:3], 3)
        j = np.tile(mesh.elem[:, 0:3], 3).flatten()
        v = np.repeat(mesh.area, 9) / 12 * np.tile(np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]).flatten(),
                                                   numelem) * np.repeat(sigma, 9) / np.repeat(mesh.depth, 9)

        mfem = sp.csr_matrix((v, (i - 1, j - 1)), shape=(numnode, numnode))
        return mfem

    @staticmethod
    def prb_mate_2_elem(prb, mateprop):
        prop = mateprop

        if prop == 'nu':
            mateprop = np.array(
                [[795774.71545948, 795774.71545948], [795774.71545948, 795774.71545948], [795.77471546, 795.77471546]])
        elif prop == 'sigma':
            mateprop = np.array(
                [[0], [0], [10000000]])
        else:
            print("error")
            return

        rg = prb.mesh.elem[:, 3]
        mt = np.array(prb.region[rg - 1, 2], dtype=int)
        elemprop = mateprop[mt - 1, :]

        return elemprop
