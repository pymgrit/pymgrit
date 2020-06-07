"""
Helper functions for the induction
machine model "im_3kW". (https://gitlab.onelab.info/doc/models/-/wikis/Electric-machines)
"""

from typing import Tuple, List, Dict

import numpy as np
import scipy.spatial.qhull as qhull


def is_numeric(obj) -> bool:
    """
    Test if obj is numeric

    :param obj: test object
    :return: is numeric
    """
    try:
        obj + 0
        return True
    except TypeError:
        return False


def get_preresolution(file: str) -> int:
    """
    Read pre file and returns number of unknowns

    :param file: pre file
    :return: number of unknowns
    """
    with open(file) as f:
        content = f.readlines()
    ind = [idx for idx, s in enumerate(content) if '$DofData' in s][0]
    tmp = content[ind + 5].split()[-1]

    return int(tmp)


def set_resolution(file: str, t_start: float, u_start: np.ndarray, num_dofs: int) -> None:
    """
    Create resolution file

    :param file: file
    :param t_start: time associated with the input approximate solution u_start
    :param u_start: approximate solution for the input time t_start
    :param num_dofs: number of unknowns
    """
    dofpos = np.cumsum([0, num_dofs])
    com_str = ['$ResFormat /* GetDP 2.10.0, ascii */', '1.1 0', '$EndResFormat']

    for j in range(np.size(t_start)):
        for k in range(np.size(num_dofs)):
            com_str.append('$Solution  /* DofData #' + str(k) + ' */')
            com_str.append(str(k) + ' ' + str(t_start) + ' 0 ' + str(j))
            y = u_start[dofpos[k]: dofpos[k + 1]]
            com_str.append("\n".join(" ".join(map(str, line)) for line in np.vstack((np.real(y), np.imag(u_start))).T))
            com_str.append('$EndSolution\n')

    with open(file, "w") as fid:
        fid.write("\n".join(com_str))


def get_values_from(file: str) -> np.ndarray:
    """
    Read values from file

    :param file: result file
    :return: result value(s)
    """
    val = []
    with open(file) as fobj:
        for line in fobj:
            row = line.split()
            val.append(row[-1])
    return np.array(val, dtype=float)


def getdp_read_resolution(file: str, num_dofs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read unknown values from file

    :param file: result file
    :param num_dofs: number of unknowns
    :return: timepoint(s) and value(s)
    """
    # init solution vector, may contain several dofdata sets
    x = np.zeros((0, np.sum(num_dofs)))
    # init vector of time steps
    t = np.zeros(0)
    # init vector of time step numbers
    j = 0
    oldstep = 0
    # get positions of dofdata in x vector
    dofpos = np.cumsum([0, num_dofs])

    with open(file) as f:
        content = f.readlines()

    idx = 0
    while idx < len(content):
        if content[idx].find('$Solution') != -1:
            idx = idx + 1
            line = content[idx]
            idx = idx + 1
            tmp = line.split()
            tmp = [int(tmp[0]), float(tmp[1]), float(tmp[2]), int(tmp[3])]
            if oldstep < 1 + tmp[3]:
                j = j + 1
                oldstep = 1 + tmp[3]
                x = np.vstack((x, np.zeros((1, np.sum(num_dofs)))))
                t = np.hstack((t, 0))
            elif oldstep > 1 + tmp[3]:
                raise Exception('Exception reading file #s. time step #d is stored after #d', file,
                                tmp[3], oldstep - 1)
            k = 1 + tmp[0]
            t[j - 1] = tmp[1]
            # read complex dofdata set into solution vector
            xtmp = content[idx:idx + num_dofs]
            xtmp = np.array([list(map(float, s.split())) for s in xtmp])
            x[j - 1, dofpos[k - 1]:dofpos[k] + 1] = (xtmp[:, 0] + np.imag(xtmp[:, 1])).T
            idx = idx + num_dofs

        elif content[idx].find('$ResFormat') != -1:
            idx = idx + 1
            if not content[idx][0:3] == '1.1':
                raise Exception('Unknown file format version')
        else:
            idx = idx + 1

    if np.max(np.isnan(x)) or np.max(np.isnan(t)):
        raise Exception('getdp_read_resolution: file contains NaN | timepoint:', t)

    return t, x


def pre_file(file: str) -> Tuple[Dict, Dict, List]:
    """
    Read pre file and return mapping between nodes

    :param file: pre file
    :return: mapping between unknowns and grid points
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


def compute_data(pre: str, msh: str, new_unknown_start: int, inner_r: float = 0.04568666666666668) -> Dict:
    """
    Compute grid information

    :param pre: Pre file of mesh
    :param msh: Mesh
    :param new_unknown_start:
    :param inner_r: Radius machine
    :return:
    """
    cor_to_un, un_to_cor, boundary = pre_file(pre)
    nodes, nodes_r = get_nodes(msh)
    lines, elements, lines_r, elements_r = get_elements(msh)

    tmp = get_arrays(nodes, lines, elements, inner_r, un_to_cor, boundary, new_unknown_start)
    data = {'nodes': nodes, 'lines': lines, 'elements': elements, 'elementsR': elements_r, 'linesR': lines_r,
            'nodesR': nodes_r, 'corToUn': cor_to_un, 'unToCor': un_to_cor, 'boundary': boundary,
            'pointsCom': tmp['pointsCom'], 'pointsBou': tmp['pointsBou'], 'pointsInner': tmp['pointsInner'],
            'pointsBouInner': tmp['pointsBouInner'], 'elecom': tmp['elecom'], 'unknown': tmp['unknown'],
            'unknownCom': tmp['unknownCom'], 'indNodesToI': tmp['ind'],
            'boundaryNodes': tmp['boundaryNodes'], 'unknownComInner': tmp['unknownComInner'],
            'unknownComOuter': tmp['unknownComOuter'], 'unknownInner': tmp['unknownInner'],
            'unknownOuter': tmp['unknownOuter'], 'pointsOuter': tmp['pointsOuter'],
            'pointsBouOuter': tmp['pointsBouOuter'], 'mappingInnerToUnknown': tmp['mappingInnerToUnknown'],
            'mappingOuterToUnknown': tmp['mappingOuterToUnknown'], 'unknownNewInner': tmp['unknownNewInner'],
            'unknownNewOuter': tmp['unknownNewOuter'], 'mappingInnerToUnknownNew': tmp['mappingInnerToUnknownNew'],
            'mappingOuterToUnknownNew': tmp['mappingOuterToUnknownNew'], 'unknownNew': tmp['unknownNew']}
    return data

def check_version(msh_file:str):

    with open(msh_file) as f:
        content = f.readlines()

    if content[1].split()[0] != '4':
        raise Exception('Unsupported msh version. Required version: 4')

def compute_mesh_transfer(values: np.ndarray, vtx: np.ndarray, wts: np.ndarray, dif: int, dif2: int,
                          fill_value: float = np.nan) -> np.ndarray:
    """
    Compute mesh transfer

    :param values: vector to transform
    :param vtx: vertices
    :param wts: weights
    :param dif: difference for boundary conditions
    :param dif2: difference for boundary conditions
    :param fill_value: fill value
    :return: input values transfered to another grid
    """
    work = np.append(values, np.zeros(dif))
    ret = np.einsum('nj,nj->n', np.take(work, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    ret = ret[:(np.size(ret) - dif2)]
    return ret


def get_nodes(file: str) -> Tuple[Dict, Dict]:
    """
    Get nodes from file

    :param file: Mesh file
    :return: Grid points
    """
    with open(file) as f:
        content = f.readlines()

    start = 0
    end = 0
    for i in range(len(content)):
        if content[i] == '$Nodes\n':
            start = i
        if content[i] == '$EndNodes\n':
            end = i

    nodes = content[start + 2:end]

    node_dict = {}
    point_to_node = {}
    for node in nodes:
        row = node.split()
        if len(row) > 1 and row[1] != '0' and row[1] != '1' and row[1] != '2':
            node_dict[row[0]] = np.array([float(row[1]), float(row[2])])
            point_to_node[row[1] + ' ' + row[2]] = row[0]

    return node_dict, point_to_node


def get_elements(file: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Get elements from file

    :param file: mesh file
    :return: elements
    """
    with open(file) as f:
        content = f.readlines()

    start_ele = 0
    end_ele = 0
    for i in range(len(content)):
        if content[i] == '$EndElements\n':
            end_ele = i
        if content[i] == '$Elements\n':
            start_ele = i

    ele = content[start_ele + 2:end_ele]
    ele_new_line = []
    ele_new_triangle = []
    i = 0
    while i < len(ele):
        num = int(ele[i].split()[-1])
        if len(ele[i + 1].split()) == 3:
            ele_new_line = ele_new_line + ele[i + 1:i + num + 1]
        elif len(ele[i + 1].split()) == 4:
            ele_new_triangle = ele_new_triangle + ele[i + 1:i + num + 1]
        i = i + num + 1

    ele_line_dict = {}
    ele_triangle_dict = {}
    ele_line_dict_reverse = {}
    ele_triangle_dict_reverse = {}

    for elem in ele_new_line:
        row = elem.split()
        ele_line_dict[row[0]] = np.array([row[1], row[2]])
        ele_line_dict_reverse[row[1] + ' ' + row[2]] = row[0]

    for elem in ele_new_triangle:
        row = elem.split()
        ele_triangle_dict[row[0]] = np.array([row[1], row[2], row[3]])
        ele_triangle_dict_reverse[row[1] + ' ' + row[2] + ' ' + row[3]] = row[0]

    return ele_line_dict, ele_triangle_dict, ele_line_dict_reverse, ele_triangle_dict_reverse


def cart2pol(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Transform Cartesian coordinates to polar

    :param x: x-coordinates
    :param y: y-coordinates
    :return: polar coordinates
    """
    return (x ** 2 + y ** 2) ** .5


def get_arrays(nodes_dict: Dict, lines_dict: Dict, elements_dict: Dict, inner_r: float,
               unknown_to_cor: Dict, boundary: List, new_start: int = 0) -> Dict:
    """
    Compute mapping between grids

    :param nodes_dict: grid points
    :param lines_dict: lines
    :param elements_dict: elements
    :param inner_r: radius machine
    :param unknown_to_cor: mapping between unknown and grid points
    :param boundary: boundary elements
    :param new_start: new unknowns
    :return: grid information
    """
    points_com = np.zeros((len(nodes_dict), 2))
    ind = {}

    i = 0
    for key, val in nodes_dict.items():
        points_com[i, 0] = val[0]
        points_com[i, 1] = val[1]
        ind[key] = i
        i = i + 1

    boundary_nodes = set([])
    for key, val in lines_dict.items():
        boundary_nodes.add(val[0])
        boundary_nodes.add(val[1])

    boundary_nodes = list(boundary_nodes)
    boundary_nodes.sort()

    i = 0
    points_bou = np.zeros((len(boundary_nodes), 2))
    for node in boundary_nodes:
        points_bou[i] = nodes_dict[node]
        i = i + 1

    elecom = np.zeros((len(elements_dict), 3), dtype=int)
    i = 0
    for key, val in elements_dict.items():
        elecom[i, 0] = ind[val[0]]
        elecom[i, 1] = ind[val[1]]
        elecom[i, 2] = ind[val[2]]
        i = i + 1

    unknown = np.zeros((len(unknown_to_cor), 2))
    unknown_com = np.zeros((len(unknown_to_cor) + len(boundary), 2))
    i = 0
    for key, val in unknown_to_cor.items():
        node = nodes_dict[val]
        unknown[i, 0] = node[0]
        unknown[i, 1] = node[1]
        unknown_com[i, 0] = node[0]
        unknown_com[i, 1] = node[1]
        i = i + 1
    for elem in boundary:
        node = nodes_dict[elem]
        unknown_com[i, 0] = node[0]
        unknown_com[i, 1] = node[1]
        i = i + 1

    unknown_new = np.copy(unknown[new_start:, :])

    r = cart2pol(points_com[:, 0], points_com[:, 1])
    inner = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
    points_inner = points_com[inner]
    outer = np.where(abs(r) > abs(inner_r) - 1e-9)[0]
    points_outer = points_com[outer]

    r = cart2pol(points_bou[:, 0], points_bou[:, 1])
    inner_boundary_nodes = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
    points_bou_inner = points_bou[inner_boundary_nodes]
    outer_boundary_nodes = np.where(abs(r) > abs(inner_r) + 1e-7)[0]
    points_bou_outer = points_bou[outer_boundary_nodes]

    r = cart2pol(unknown_com[:, 0], unknown_com[:, 1])
    inner = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
    unknown_com_inner = unknown_com[inner]
    outer = np.where(abs(r) > abs(inner_r) + 1e-7)[0]
    unknown_com_outer = unknown_com[outer]

    r = cart2pol(unknown[:, 0], unknown[:, 1])
    inner = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
    unknown_inner = unknown[inner]
    outer = np.where(abs(r) > abs(inner_r) + 1e-7)[0]
    unknown_outer = unknown[outer]

    r = cart2pol(unknown_new[:, 0], unknown_new[:, 1])
    inner = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
    unknown_new_inner = unknown_new[inner]
    outer = np.where(abs(r) > abs(inner_r) + 1e-7)[0]
    unknown_new_outer = unknown_new[outer]

    mapping_inner_to_unknown = np.zeros(len(unknown_inner), dtype=int)
    mapping_outer_to_unknown = np.zeros(len(unknown_outer), dtype=int)

    mapping_inner_to_unknown_new = np.zeros(len(unknown_new_inner), dtype=int)
    mapping_outer_to_unknown_new = np.zeros(len(unknown_new_outer), dtype=int)

    k = 0
    s = 0
    for i in range(np.size(unknown_new, 0)):
        if unknown_new[i] in unknown_new_inner:
            mapping_inner_to_unknown_new[k] = i
            k = k + 1
        elif unknown_new[i] in unknown_new_outer:
            mapping_outer_to_unknown_new[s] = i
            s = s + 1

    k = 0
    i = 0
    s = 0
    for key, val in unknown_to_cor.items():
        node = nodes_dict[val]
        if node in unknown_inner:
            mapping_inner_to_unknown[k] = i
            k = k + 1
        if node in unknown_outer:
            mapping_outer_to_unknown[s] = i
            s = s + 1
        i = i + 1

    ret_dict = {
        'pointsCom': points_com,
        'pointsBou': points_bou,
        'pointsInner': points_inner,
        'pointsBouInner': points_bou_inner,
        'elecom': elecom,
        'unknown': unknown,
        'unknownCom': unknown_com,
        'ind': ind,
        'boundaryNodes': boundary_nodes,
        'pointsOuter': points_outer,
        'pointsBouOuter': points_bou_outer,
        'unknownComInner': unknown_com_inner,
        'unknownComOuter': unknown_com_outer,
        'unknownInner': unknown_inner,
        'unknownOuter': unknown_outer,
        'mappingInnerToUnknown': mapping_inner_to_unknown,
        'mappingOuterToUnknown': mapping_outer_to_unknown,
        'unknownNewInner': unknown_new_inner,
        'unknownNewOuter': unknown_new_outer,
        'mappingInnerToUnknownNew': mapping_inner_to_unknown_new,
        'mappingOuterToUnknownNew': mapping_outer_to_unknown_new,
        'unknownNew': unknown_new
    }
    return ret_dict


def interpolation_factors(data_coarse: Dict, data_fine: Dict) -> Dict:
    """
    Compute the interpolation factor for each point by two given grids

    :param data_coarse:
    :param data_fine:
    :return:
    """
    # vtxCom, wtsCom = interp_weights(data_coarse['unknownCom'], data_fine['unknown'][len(data_coarse['unToCor']):])
    vtx_inner, wts_inner = interp_weights(data_coarse['unknownComInner'], data_fine['unknownNewInner'])
    vtx_outer, wts_outer = interp_weights(data_coarse['unknownComOuter'], data_fine['unknownNewOuter'])

    add_bound_inner = np.size(data_coarse['unknownComInner'], 0) - np.size(data_coarse['unknownInner'], 0)
    add_bound_outer = np.size(data_coarse['unknownComOuter'], 0) - np.size(data_coarse['unknownOuter'], 0)
    size_lvl_stop = len(data_fine['corToUn'])
    size_lvl_start = len(data_coarse['corToUn'])
    mapping_inner = data_coarse['mappingInnerToUnknown']
    mapping_outer = data_coarse['mappingOuterToUnknown']
    mapping_inner_new = data_fine['mappingInnerToUnknownNew']
    mapping_outer_new = data_fine['mappingOuterToUnknownNew']

    ret_dict = {
        'vtxInner': vtx_inner,
        'wtsInner': wts_inner,
        'vtxOuter': vtx_outer,
        'wtsOuter': wts_outer,
        'addBoundInner': add_bound_inner,
        'addBoundOuter': add_bound_outer,
        'sizeLvlStop': size_lvl_stop,
        'sizeLvlStart': size_lvl_start,
        'mappingInner': mapping_inner,
        'mappingOuter': mapping_outer,
        'mappingInnerNew': mapping_inner_new,
        'mappingOuterNew': mapping_outer_new
    }

    return ret_dict


def interp_weights(xyz: np.ndarray, uvw: np.ndarray, d: int = 2, tol: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolation between two grids

    :param xyz: coarse grid points
    :param uvw: fine grid points
    :param d: dimensions
    :param tol: tolerance
    :return: vertices and weights
    """
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw, tol=tol)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    wts[wts < 0] = 0
    return vertices, wts
