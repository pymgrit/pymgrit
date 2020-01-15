import unittest
import numpy as np
import scipy.io as sio
import os.path

from pymgrit.cable_current_driven import cable_current_driven


class TestCableCurrentDriven(unittest.TestCase):
    problem_directory = None
    current_directory = None

    def setUp(self):
        self.problem_directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..',
            '..',
            'src',
            'pymgrit',
            'cable_current_driven',
            'problems'
        )

        self.current_directory = os.path.dirname(os.path.realpath(__file__))


    def test_nlin_initialise(self):
        # prb = sio.loadmat(self.problem_directory + '/cable_61.mat', struct_as_record=False, squeeze_me=True)['cable_61']

        bh = np.loadtxt(self.problem_directory + '/BH.txt')

        bchar = bh[:, 0]
        hchar = bh[:, 1]
        nlin = cable_current_driven.CableCurrentDriven.nlin_initialise(bchar, hchar)

        b = np.array([0.01, 0.02, 0.035, 0.05, 0.07, 0.107, 0.14, 0.2, 0.33, 1.02, 1.25, 1.35,
                      1.4, 1.43, 1.45, 1.47, 1.48, 1.495, 1.58, 1.64, 1.675, 1.71, 1.73, 1.76,
                      1.78, 1.795, 1.81, 1.95, 2.03, 2.07, 2.18, 2.25])
        h = np.array([9.9997048e+00, 1.9999410e+01, 2.9999114e+01, 3.9998819e+01, 4.9998524e+01,
                      5.9998229e+01, 6.9997933e+01, 7.9997638e+01, 9.9997048e+01, 1.9999410e+02,
                      2.9999114e+02, 3.9998819e+02, 4.9998524e+02, 5.9998229e+02, 6.9997933e+02,
                      7.9997638e+02, 8.9997343e+02, 9.9997048e+02, 1.9999410e+03, 2.9999114e+03,
                      3.9998819e+03, 4.9998524e+03, 5.9998229e+03, 6.9997933e+03, 7.9997638e+03,
                      8.9997343e+03, 9.9997048e+03, 1.9999410e+04, 2.9999114e+04, 3.9788735e+04,
                      7.9577469e+04, 1.1140846e+05])
        bmin = 0.01
        bmax = 2.25
        hmin = 9.9997048
        hmax = 111408.46
        initialslope = np.array([999.97048])
        finalslope = np.array([454728.44285714])
        finalcoercitivity = np.array([-911730.53642857])
        finalremanence = np.array([2.00500002])
        final_wmagn = np.array([20056.28209796])

        np.testing.assert_equal(bmin, nlin['Bmin'])
        np.testing.assert_equal(bmax, nlin['Bmax'])
        np.testing.assert_equal(hmin, nlin['Hmin'])
        np.testing.assert_equal(hmax, nlin['Hmax'])
        np.testing.assert_almost_equal(b, nlin['B'])
        np.testing.assert_almost_equal(h, nlin['H'])
        np.testing.assert_almost_equal(initialslope, nlin['initialslope'])
        np.testing.assert_almost_equal(finalslope, nlin['finalslope'])
        np.testing.assert_almost_equal(finalcoercitivity, nlin['finalcoercitivity'])
        np.testing.assert_almost_equal(finalremanence, nlin['finalremanence'])
        np.testing.assert_almost_equal(final_wmagn, nlin['finalWmagn'])

    def test_prb_mate_2_elem(self):
        prb = sio.loadmat(self.problem_directory + '/cable_61.mat', struct_as_record=False, squeeze_me=True)['cable_61']

        sigmaelem = cable_current_driven.CableCurrentDriven.prb_mate_2_elem(prb, 'sigma').transpose()[0]

        test_sigmaelem = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000,
                                   10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000],
                                  dtype=int)

        np.testing.assert_equal(test_sigmaelem, sigmaelem)

    def test_prb_mate_2_elem_nu(self):
        prb = sio.loadmat(self.problem_directory + '/cable_61.mat', struct_as_record=False, squeeze_me=True)['cable_61']

        nuelem = cable_current_driven.CableCurrentDriven.prb_mate_2_elem(prb, 'nu').transpose()

        test_nuelem = np.ones((2, 96))
        test_nuelem[:, :24] *= 795774.71545948
        test_nuelem[:, 24:] *= 795.77471546

        np.testing.assert_equal(test_nuelem, nuelem)

    def test_current_pstr(self):
        prb = sio.loadmat(self.problem_directory + '/cable_61.mat', struct_as_record=False, squeeze_me=True)['cable_61']

        pfem = cable_current_driven.CableCurrentDriven.current_pstr(prb)

        test_pfem = np.load(self.current_directory + '/pfem.npy', allow_pickle=True).item()

        np.testing.assert_equal(test_pfem.toarray(), pfem.toarray())

    def test_edgemass_ll(self):
        prb = sio.loadmat(self.problem_directory + '/cable_61.mat', struct_as_record=False, squeeze_me=True)['cable_61']

        sigmaelem = cable_current_driven.CableCurrentDriven.prb_mate_2_elem(prb, 'sigma').transpose()[0]

        mfem = cable_current_driven.CableCurrentDriven.edgemass_ll(prb.mesh, sigmaelem)

        test_mfem = np.load(self.current_directory + '/mfem.npy', allow_pickle=True).item()

        np.testing.assert_almost_equal(test_mfem.toarray(), mfem.toarray())

    def test_curlcurl_ll(self):
        prb = sio.loadmat(self.problem_directory + '/cable_61.mat', struct_as_record=False, squeeze_me=True)['cable_61']

        nuelem = cable_current_driven.CableCurrentDriven.prb_mate_2_elem(prb, 'nu').transpose()

        kfem = cable_current_driven.CableCurrentDriven.curlcurl_ll(prb.mesh, nuelem)

        test_kfem = np.load(self.current_directory + '/kfem.npy', allow_pickle=True).item()

        np.testing.assert_almost_equal(test_kfem.toarray(), kfem.toarray())

    def test_cart2pol(self):
        prb = sio.loadmat(self.problem_directory + '/cable_61.mat', struct_as_record=False, squeeze_me=True)['cable_61']

        x = prb.mesh.node[:, 0]
        y = prb.mesh.node[:, 1]
        r = cable_current_driven.CableCurrentDriven.cart2pol(x, y)

        test_r = np.array([0., 0.00254, 0.00254, 0.00254, 0.00254, 0.00254, 0.00254, 0.0127, 0.0127,
                           0.0127, 0.0127, 0.0127, 0.0127, 0.0127, 0.0127, 0.0127, 0.0127, 0.0127,
                           0.0127, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905,
                           0.01905, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905, 0.01905,
                           0.01905, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254,
                           0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254,
                           0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254, 0.0254, ])
        np.testing.assert_almost_equal(test_r, r)

    def test_save_divide(self):
        x = np.zeros(3198)
        y = np.zeros(3198)
        value = np.ones(1) * 999.97048

        test_z = np.ones(3198) * 999.97048

        z = cable_current_driven.CableCurrentDriven.save_divide(x, y, value)

        np.testing.assert_equal(test_z, z)

    def test_pyth(self):
        test_bm = (np.array([1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356,
                             1.41421356, 1.41421356, 1.41421356, 1.41421356, 1.41421356]),
                   np.array([[0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678,
                              0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678],
                             [0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678,
                              0.70710678, 0.70710678, 0.70710678, 0.70710678, 0.70710678]]))

        bm = cable_current_driven.CableCurrentDriven.pyth(np.ones((2, 10)))

        np.testing.assert_almost_equal(test_bm[0], bm[0])
        np.testing.assert_almost_equal(test_bm[1], bm[1])

    def test_utl_pwm(self):
        f = cable_current_driven.CableCurrentDriven.utl_pwm(0.005, 50, 400)
        np.testing.assert_equal(1, f)

    def test_phi_linear_pwm(self):
        t_start = 0
        t_stop = 0.005
        u = np.zeros(37)

        cable_2 = cable_current_driven.CableCurrentDriven(nonlinear=False, pwm=True, name='cable_61',
                                                          t_start=0, t_stop=0.02, nt=5)

        res = cable_2.phi_linear(t_start=t_start, t_stop=t_stop, vinit=u)

        test_res = np.array(
            [0.00031907, 0.00030698, 0.00030698, 0.00030698, 0.00030698, 0.00030698, 0.00030698, 0.00028019, 0.00028082,
             0.00028019, 0.00028082, 0.00028019, 0.00028082, 0.00028019, 2.80818231e-04, 2.80188467e-04, 2.80818226e-04,
             2.80188467e-04, 2.80818231e-04, - 1.83323852e-05, - 7.09702331e-05, - 7.09702406e-05, - 1.83323673e-05,
             - 7.09702787e-05, - 7.09702787e-05, - 1.83323673e-05, - 7.09702406e-05, - 7.09702331e-05, -1.83323852e-05,
             - 7.09702331e-05, - 7.09702406e-05, - 1.83323673e-05, - 7.09702787e-05, - 7.09702787e-05, - 1.83323673e-05,
             - 7.09702406e-05, - 7.09702331e-05])

        np.testing.assert_almost_equal(test_res, res)

    def test_phi_linear_sin(self):
        t_start = 0
        t_stop = 0.005
        u = np.zeros(37)

        cable_2 = cable_current_driven.CableCurrentDriven(nonlinear=False, pwm=False, name='cable_61',
                                                          t_start=0, t_stop=0.02, nt=5)

        res = cable_2.phi_linear(t_start=t_start, t_stop=t_stop, vinit=u)

        test_res = np.array(
            [0.00031907, 0.00030698, 0.00030698, 0.00030698, 0.00030698, 0.00030698,
             0.00030698, 0.00028019, 0.00028082, 0.00028019, 0.00028082, 0.00028019,
             0.00028082, 0.00028019, 0.00028082, 0.00028019, 0.00028082, 0.00028019,
             2.80818231e-04, - 1.83323852e-05, - 7.09702331e-05, - 7.09702406e-05,
             - 1.83323673e-05, - 7.09702787e-05, - 7.09702787e-05, - 1.83323673e-05,
             - 7.09702406e-05, - 7.09702331e-05, - 1.83323852e-05, - 7.09702331e-05,
             - 7.09702406e-05, - 1.83323673e-05, - 7.09702787e-05,
             -7.09702787e-05, - 1.83323673e-05, - 7.09702406e-05, - 7.09702331e-05])
        np.testing.assert_almost_equal(test_res, res)

    def test_step_newton(self):
        t_start = 0
        t_stop = 0.005
        u = np.zeros(37)

        cable_2 = cable_current_driven.CableCurrentDriven(nonlinear=True, pwm=False, name='cable_61',
                                                          t_start=0, t_stop=0.02, nt=5)

        res = cable_2.newton(vinit=u, tstart=t_start, tstop=t_stop)

        test_res = np.array([0.00031899, 0.0003069, 0.0003069, 0.0003069, 0.0003069, 0.0003069,
                             0.0003069, 0.00028011, 0.00028074, 0.00028011, 0.00028074, 0.00028011,
                             0.00028074, 0.00028011, 0.00028074,
                             2.80106736e-04, 2.80738936e-04, 2.80106736e-04, 2.80738941e-04,
                             - 1.85169630e-05, - 7.05516737e-05, - 7.05516798e-05, - 1.85169460e-05,
                             - 7.05517172e-05, - 7.05517172e-05, - 1.85169460e-05, - 7.05516798e-05,
                             - 7.05516737e-05, - 1.85169630e-05, - 7.05516737e-05,
                             -7.05516798e-05, -1.85169460e-05, -7.05517172e-05, -7.05517172e-05,
                             -1.85169460e-05, -7.05516798e-05, -7.05516737e-05, ])

        np.testing.assert_almost_equal(test_res, res)

if __name__ == '__main__':
    unittest.main()
