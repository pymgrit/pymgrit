from application import application
import scipy.sparse as sp
from scipy import interpolate
from scipy import linalg as la
from scipy.sparse.linalg import spsolve
import numpy as np
import scipy.io as sio
from scipy import sparse
import os
import scipy.spatial.qhull as qhull


class CableCurrentDriven(application.Application):
    """
    """

    def __init__(self, linear, pwm, name, coarse_smooth, *args, **kwargs):
        super(CableCurrentDriven, self).__init__(*args, **kwargs)
        self.linear = linear

        self.name = name
        self.pwm = pwm
        self.coarse_smooth = coarse_smooth
        self.coarse_lvl, self.prb = self.get_coarse_grids(os.getcwd() + '/cable_current_driven/problems/' + self.name)
        self.nx = np.zeros(self.coarse_lvl, dtype=int)
        self.intom = 0.0254

        for i in range(len(self.prb)):
            r2 = 1.0 * self.intom
            x = self.prb[i].mesh.node[:, 0]
            y = self.prb[i].mesh.node[:, 1]
            r = self.cart2pol(x, y)
            self.nx[i] = len(np.where(abs(r - r2) > 1e-8)[0])

    def setup(self, lvl_max, t, spatial_coarsening):
        """
        """
        pwm_tmp = np.repeat(int(self.pwm), lvl_max)
        if self.coarse_smooth:
            pwm_tmp[1:] = self.coarse_smooth

        intom = 0.0254
        tol = 1e-8
        # r0 = 0.1 * intom
        r1 = 0.5 * intom
        r2 = 1.0 * intom
        iw = 100
        f = 50
        omega = 2 * np.pi * f
        rg_fe = 3

        bh = np.loadtxt('/'.join(__file__.split('/')[:-1]) + '/problems/BH.txt')

        bchar = bh[:, 0]
        hchar = bh[:, 1]
        nlin = self.nlin_initialise(bchar, hchar)

        x = [np.zeros(0)] * lvl_max
        y = [np.zeros(0)] * lvl_max
        idxnlinelem = [np.zeros(0)] * lvl_max
        idxdof = [np.zeros(0)] * lvl_max
        msh = [None] * lvl_max
        ksh = [None] * lvl_max
        psh = [None] * lvl_max
        app = [{}] * lvl_max
        points = [np.zeros(0)] * lvl_max
        nuelem = [None] * lvl_max
        dif = [0] * lvl_max
        vtx_int = [None] * lvl_max
        wts_int = [None] * lvl_max
        vtx_res = [None] * lvl_max
        wts_res = [None] * lvl_max
        r1_max_node = [np.zeros(0)] * lvl_max
        restriction = [None] * lvl_max
        interpolation = [None] * lvl_max
        # restriction_bay = [None] * lvl_max

        i = 0
        for lvl in range(lvl_max):
            sigmaelem = self.prb_mate_2_elem(self.prb[i], 'sigma').transpose()[0]
            nuelem[lvl] = self.prb_mate_2_elem(self.prb[i], 'nu').transpose()

            elemregi = np.zeros_like(self.prb[i].mesh.elem[:, 3])
            elemregi[self.prb[i].mesh.elem[:, 3] == 1] = self.prb[i].region[0, 2]
            elemregi[self.prb[i].mesh.elem[:, 3] == 2] = self.prb[i].region[1, 2]
            elemregi[self.prb[i].mesh.elem[:, 3] == 3] = self.prb[i].region[2, 2]
            idxnlinelem[lvl] = np.where(elemregi == rg_fe)[0]

            pfem = self.current_pstr(self.prb[i])
            mfem = self.edgemass_ll(self.prb[i].mesh, sigmaelem)
            kfem = self.curlcurl_ll(self.prb[i].mesh, nuelem[lvl])

            x[lvl] = self.prb[i].mesh.node[:, 0]
            y[lvl] = self.prb[i].mesh.node[:, 1]
            r = self.cart2pol(x[lvl], y[lvl])
            idxdof[lvl] = np.where(abs(r - r2) > tol)
            r1_max_node[lvl] = np.max(np.where(abs(r - r1) <= tol))

            points[lvl] = np.vstack((x[lvl], y[lvl])).transpose()
            dif[lvl] = np.size(points[lvl], 0) - np.size(idxdof[lvl])

            msh[lvl] = mfem[0:np.size(idxdof[lvl]), 0:np.size(idxdof[lvl])]
            ksh[lvl] = kfem[0:np.size(idxdof[lvl]), 0:np.size(idxdof[lvl])]
            psh[lvl] = pfem[idxdof[lvl], 0]
            if spatial_coarsening[lvl]:
                i = i + 1

            # TODO
        #for lvl in range(lvl_max - 1):
        #    vtx_int[lvl], wts_int[lvl] = self.interp_weights(points[lvl + 1], points[lvl], d=2)
        #    vtx_res[lvl], wts_res[lvl] = self.interp_weights(points[lvl], points[i + 1], d=2)
        #    wts_int[lvl][np.size(idxdof[lvl]):] = 0
        #    vtx_int[lvl][np.size(idxdof[lvl]):] = 0

        #    interpolation[lvl] = sparse.lil_matrix((np.size(points[lvl], 0), np.size(points[lvl + 1], 0)), dtype=float)
        #    for j in range(np.size(points[lvl], 0)):
        #        interpolation[lvl][j, vtx_int[lvl][j]] = wts_int[lvl][j]
        #    restriction[lvl] = sparse.csr_matrix(
        #        (1 * interpolation[lvl].transpose())[:np.size(idxdof[i + 1]), :np.size(idxdof[lvl])])
        #    interpolation[lvl] = sparse.csr_matrix(interpolation[lvl][:np.size(idxdof[lvl]), :np.size(idxdof[i + 1])])

        #    restriction_bay[lvl] = sparse.lil_matrix((np.size(points[i + 1], 0), np.size(points[lvl], 0)), dtype=float)
        #    for j in range(np.size(points[i + 1], 0)):
        #        restriction_bay[lvl][j, vtx_res[lvl][j]] = wts_res[lvl][j]
        #    restriction_bay[lvl] = sparse.csr_matrix(
        #        restriction_bay[lvl][:np.size(idxdof[i + 1]), :np.size(idxdof[lvl])])

        for lvl in range(lvl_max):
            app[lvl] = {'Msh': msh[lvl], 'nu': nuelem[lvl], 'Psh': psh[lvl], 'idxdof': idxdof[lvl],
                        'idxnlinelem': idxnlinelem[lvl], 'Iw': iw, 'f': f, 'omega': omega, 'nlin': nlin,
                        'mesh': self.prb[i].mesh,
                        'vtx_int': vtx_int[lvl], 'wts_int': wts_int[lvl], 'vtx_res': vtx_res[lvl],
                        "wts_res": wts_res[lvl],
                        'dif': dif[lvl], 'Ksh': ksh[lvl], 'r': restriction[lvl], 'P': interpolation[lvl], 'x': x[lvl],
                        'y': y[lvl], 'pwm': pwm_tmp[lvl], 'linear': self.linear}
            if spatial_coarsening[lvl]:
                i = i + 1

        return app

    def initial_value(self):
        """

        :rtype: object
        """
        return np.zeros(self.nx[0])

    def phi(self, u_start, t_start, t_stop, app):
        if app['linear']:
            return self.phi_linear(t_start, t_stop, u_start, app)
        else:
            return self.newton(t_start, t_stop, u_start, app)

    @staticmethod
    def get_coarse_grids(name):
        count = 0
        prb = []
        for element in list(sio.loadmat(name, struct_as_record=False, squeeze_me=True).values()):
            if isinstance(element, sio.matlab.mio5_params.mat_struct):
                prb = prb + [element]
                count = count + 1
        return count, prb

    def phi_linear(self, t_start, t_stop, vinit, app):
        vstepsize = t_stop - t_start

        if app['pwm']:
            f = app['Msh'].dot(vinit) / vstepsize + \
                (app['Psh'] * app['Iw'] * self.utl_pwm(t_stop, 50)).toarray()[0]
        else:
            f = app['Msh'].dot(vinit) / vstepsize + \
                (app['Psh'] * app['Iw'] * np.sin(app['omega'] * t_stop)).toarray()[0]

        return spsolve(app['Msh'] / vstepsize + app['Ksh'], f)

    def cpwm(self, t, size):
        tmp = np.zeros(size)
        tmp[-1] = 1
        return (1 / 4) * (-self.utl_pwm(t, 50, 200) * tmp)

    @staticmethod
    def csin(t, size):
        tmp = np.zeros(size)
        tmp[-1] = 1
        return (1 / 4) * (-np.sin(2 * np.pi * 50 * t) * tmp)

    @staticmethod
    def utl_pwm(t, freq, teeth=1100):
        # sawfish pattern with higher frequency
        saw = t * teeth * freq - np.floor(t * teeth * freq)

        # plain sine wave
        sine = np.sin(freq * t * (2 * np.pi))

        # pwm signal by comparison
        pwm = np.sign(sine) * (saw - abs(sine) < 0)

        return pwm

    def newton(self, tstart, tstop, vinit, app):
        max_newton_iterations = app['max_ne_it']
        newton_tol = app['ne_tol']
        pwm = app['pwm']
        xold = np.copy(vinit)
        xnew = np.copy(vinit)

        vmass = sparse.hstack(
            (sparse.vstack((app['Msh'], -app['Psh'])), np.zeros(app['Psh'].shape[1] + 1)[np.newaxis].T)) / (
                        tstop - tstart)

        def f(x):
            return vmass.dot(x - xold) + self.eddy_current_rhs(app['mesh'], app['nu'], app['idxnlinelem'],
                                                               app['idxdof'], tstop, x, app['Psh'], app['Iw'],
                                                               app['omega'], app['nlin'], pwm)

        def j(x):
            return vmass - self.eddy_current_jac(app['mesh'], app['nu'], app['idxnlinelem'], app['idxdof'], x,
                                                 app['nlin'])

        f_value = f(xnew)
        f_norm = la.norm(f_value, np.inf)  # l2 norm of vector
        iteration_counter = 0
        while abs(f_norm) > newton_tol and iteration_counter < max_newton_iterations:
            delta = spsolve(j(xnew), -f_value)
            xnew = xnew + delta
            f_value = f(xnew)
            f_norm = la.norm(f_value, np.inf)
            iteration_counter += 1
            # print(iteration_counter)

        return xnew

    def eddy_current_rhs(self, mesh, nu, idxnlinelem, idxdof, t, ush, psh, iw, omega, nlin, pwm):
        #todo
        iw + omega

        a = np.zeros(np.size(mesh.node, 0))
        a[idxdof] = ush[:-1]
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
            c = self.cpwm(t, psh.shape[1] + 1)
        else:
            c = self.csin(t, psh.shape[1] + 1)
        b = sparse.hstack((sparse.vstack((kfem[0:np.size(idxdof), 0:np.size(idxdof)], np.zeros(psh.shape[1]))), tmp))
        rhs = b.dot(ush) - c
        return rhs

    def eddy_current_jac(self, mesh, nu, idxnlinelem, idxdof, ush, nlin):
        a = np.zeros(np.size(mesh.node, 0))
        a[idxdof] = ush[:-1]
        b = self.curl(mesh, a)
        bred = b[:, idxnlinelem]
        hred, nured, nudred, dnud_b2red = self.nlin_evaluate(nlin, bred)
        nu[:, idxnlinelem] = np.vstack((nured, nured))
        dnud_b2 = np.zeros(np.size(b, 1))
        dnud_b2[idxnlinelem] = dnud_b2red

        kfem = self.curlcurl_ll_nonlinear(mesh, b, nu, dnud_b2)  # , Hc)

        ksh = kfem[0:np.size(idxdof), 0:np.size(idxdof)]
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
        idxleft = (bm < nlin['Bmin']).nonzero()
        idxright = (bm > nlin['Bmax']).nonzero()

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

    @staticmethod
    def interp_weights(xyz, uvw, d=2, tol=0.1):
        tri = qhull.Delaunay(xyz)
        simplex = tri.find_simplex(uvw, tol=tol)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uvw - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        wts[wts < 0] = 0
        return vertices, wts

    def restriction(self, u, app=None):
        pass

    def interpolation(self, u, app=None):
        pass

    def info(self):
        return 'cable_current_driven/t-[' + str(self.t_start) + ';' + str(self.t_end) + ']/nt-' + str(
            self.nt) + '/name-' + str(self.name) + '/pwm-' + str(self.pwm) + '/linear-' + str(self.linear) + '/'
