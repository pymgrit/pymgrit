"""
Tests vector_heat_1d_2pts
"""
import numpy as np
import os

from pymgrit.induction_machine.helper import *


def test_is_numeric():
    """
    is_numeric
    """
    np.testing.assert_equal(is_numeric(2), True)
    np.testing.assert_equal(is_numeric(2.4), True)
    np.testing.assert_equal(is_numeric('str'), False)
    np.testing.assert_equal(is_numeric(np.array([4, 5])), True)


def test_get_preresolution():
    """
    get_preresolution
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    assert 31 == get_preresolution(path+'im_3kW_test_prefile.pre')


def test_set_resolution(tmpdir):
    """
    set_resolution
    """
    file = tmpdir.join('output.txt')
    set_resolution(file=file.strpath, t_start=0, u_start=np.zeros(3), num_dofs=3)
    string_list = ['$ResFormat /* GetDP 2.10.0, ascii */', '1.1 0', '$EndResFormat', '$Solution  /* DofData #0 */',
                   '0 0 0 0', '0.0 0.0\n0.0 0.0\n0.0 0.0', '$EndSolution\n']
    assert "\n".join(string_list) == file.read()


def test_get_values_from():
    """
    get_values_from
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    np.testing.assert_almost_equal(get_values_from(path+'resJL.dat'), np.array([30.66582882392347]))


def test_getdp_read_resolution():
    """
    getdp_read_resolution
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    t, x = getdp_read_resolution(path+'im_3kW.res', 4472)

    np.testing.assert_almost_equal(np.array([0.e+00, 5.e-05]), t)
    np.testing.assert_almost_equal((2, 4472), x.shape)
    np.testing.assert_almost_equal(4.211876553442336e-07, x[1][10])


def test_pre_file():
    """
    pre_file
    """
    un_to_cor = {'9': '2', '10': '3', '11': '4', '12': '5', '13': '7', '14': '8', '15': '9', '16': '10'}
    boundary = ['11', '12']
    cor_to_un = {'2': '9', '3': '10', '4': '11', '5': '12', '7': '13', '8': '14', '9': '15', '10': '16'}

    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    assert cor_to_un == pre_file(path+'im_3kW_test_prefile.pre')[0]
    assert un_to_cor == pre_file(path+'im_3kW_test_prefile.pre')[1]
    assert boundary == pre_file(path+'im_3kW_test_prefile.pre')[2]


def test_compute_data():
    """
    compute_data
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    data = compute_data(pre=path+'im_3kW_4k.pre', msh=path+'im_3kW_4k.msh', new_unknown_start=0)
    assert len(data) == 31
    assert len(data['nodes']) == 5673
    assert len(data['lines']) == 2684


def test_compute_mesh_transfer():
    """
    compute_mesh_transfer
    """
    xyz = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    uvw = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.5, 0], [0, 0.5], [1, 0.5], [0.5, 1]])
    res_vertices, res_wts = interp_weights(xyz=xyz, uvw=uvw)

    np.testing.assert_almost_equal(np.array([1., 2., 3., 4., 2.5, 1.5]),
                                   compute_mesh_transfer(np.linspace(1, 4, 4), vtx=res_vertices, wts=res_wts, dif=0,
                                                         dif2=3))


def test_get_nodes():
    """
    get_nodes
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    node_dict, point_to_node = get_nodes(path+'im_3kW_4k.msh')
    assert 5673 == len(node_dict)
    assert 5673 == len(point_to_node)

    np.testing.assert_almost_equal(np.array([0.01563382, 0.00275666]), node_dict['11'])
    np.testing.assert_almost_equal(np.array([0.01833706, 0.04167413]), node_dict['102'])
    np.testing.assert_almost_equal(np.array([-0.04522759, 0.00646036]), node_dict['189'])

    np.testing.assert_almost_equal(np.array([0.01563382, 0.00275666]),
                                   np.fromstring(
                                       list(point_to_node.keys())[list(point_to_node.values()).index('11')],
                                       dtype=float, sep=' '))
    np.testing.assert_almost_equal(np.array([0.01833706, 0.04167413]),
                                   np.fromstring(
                                       list(point_to_node.keys())[list(point_to_node.values()).index('102')],
                                       dtype=float, sep=' '))
    np.testing.assert_almost_equal(np.array([-0.04522759, 0.00646036]),
                                   np.fromstring(
                                       list(point_to_node.keys())[list(point_to_node.values()).index('189')],
                                       dtype=float, sep=' '))


def test_get_elements():
    """
    get_elements
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    ele_line_dict, ele_triangle_dict, ele_line_dict_reverse, ele_triangle_dict_reverse = get_elements(
        path+'im_3kW_4k.msh')
    assert len(ele_line_dict) == 2684
    assert len(ele_line_dict_reverse) == 2684
    assert len(ele_triangle_dict) == 8598
    assert len(ele_triangle_dict_reverse) == 8598


def test_cart2pol():
    """
    cart2pol
    """

    np.testing.assert_almost_equal(np.array([1.41421356, 2.82842712, 4.24264069, 5.65685425, 7.07106781]),
                                   cart2pol(np.linspace(1, 5, 5), np.linspace(1, 5, 5)))


def test_interpolation_factors():
    """
    interpolation_factors
    """
    path = os.path.dirname(os.path.realpath(__file__))
    if 'induction_machine' not in path:
        path += '/induction_machine/'
    else:
        path += '/'
    data_1 = compute_data(pre=path+'im_3kW_4k.pre', msh=path+'im_3kW_4k.msh', new_unknown_start=0)
    data_2 = compute_data(pre=path+'im_3kW_4k.pre', msh=path+'im_3kW_4k.msh', new_unknown_start=0)
    values = interpolation_factors(data_coarse=data_1, data_fine=data_2)

    assert len(values) == 12
    assert len(values['vtxInner']) == 2208
    assert len(values['wtsInner']) == 2208
    assert len(values['vtxOuter']) == 2241
    assert len(values['vtxOuter']) == 2241
    assert values['addBoundInner'] == 663
    assert values['addBoundOuter'] == 76
    assert values['sizeLvlStop'] == 4449
    assert values['sizeLvlStart'] == 4449
    assert len(values['mappingInner']) == 2208
    assert len(values['mappingOuter']) == 2241
    assert len(values['mappingInnerNew']) == 2208
    assert len(values['mappingOuterNew']) == 2241


def test_interp_weights():
    """
    interp_weights
    """
    xyz = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    uvw = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.5, 0], [0, 0.5], [1, 0.5], [0.5, 1]])
    res_vertices, res_wts = interp_weights(xyz=xyz, uvw=uvw)

    vertices = [[3, 2, 0], [1, 3, 0], [3, 2, 0], [3, 2, 0], [3, 2, 0], [1, 3, 0], [3, 2, 0], [1, 3, 0], [3, 2, 0]]
    wts = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0.5, 0., 0.5], [0.5, 0., 0.5], [0., 0.5, 0.5],
           [0.5, 0.5, 0.], [0.5, 0.5, 0.]]

    np.testing.assert_almost_equal(wts, res_wts)
    np.testing.assert_almost_equal(vertices, res_vertices)
