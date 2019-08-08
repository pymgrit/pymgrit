from pymgrit.core import grid_transfer
import copy
import numpy as np
import scipy.spatial.qhull as qhull


class GridTransferMachine(grid_transfer.GridTransfer):
    """
    """

    def __init__(self, coarse_grid, fine_grid):
        """
        """
        super(GridTransferMachine, self).__init__()
        path = '/'.join(__file__.split('/')[:-1]) + '/im_3kW/'
        data_coarse_pre = path + coarse_grid + '.pre'
        data_coarse_msh = path + coarse_grid + '.msh'
        data_coarse = self.compute_data(data_coarse_pre, data_coarse_msh, 0)

        data_fine_pre = path + fine_grid + '.pre'
        data_fine_msh = path + fine_grid + '.msh'
        data_fine = self.compute_data(data_fine_pre, data_fine_msh, len(data_coarse['corToUn']))

        self.transfer_data = self.interpolation_factors(data_coarse=data_coarse, data_fine=data_fine)

    def restriction(self, u):
        ret = copy.deepcopy(u)
        ret.u_middle = ret.u_middle[:self.transfer_data['sizeLvlStart']]
        return ret

    def interpolation(self, u):
        ret = copy.deepcopy(u)

        new_middle = np.zeros(self.transfer_data['sizeLvlStop'] - self.transfer_data['sizeLvlStart'])

        new_u_inner = self.compute_mesh_transfer(
            u.u_middle[self.transfer_data['mappingInner']], self.transfer_data['vtxInner'],
            self.transfer_data['wtsInner'], self.transfer_data['addBoundInner'], 0)

        new_u_outer = self.compute_mesh_transfer(
            u.u_middle[self.transfer_data['mappingOuter']], self.transfer_data['vtxOuter'],
            self.transfer_data['wtsOuter'], self.transfer_data['addBoundOuter'], 0)
        new_middle[:len(u.u_middle)] = u.u_middle
        new_middle[self.transfer_data['mappingInnerNew']] = new_u_inner
        new_middle[self.transfer_data['mappingOuterNew']] = new_u_outer
        ret.u_middle = np.append(ret.u_middle, new_middle)

        return ret

    def compute_data(self, pre, msh, new_unknown_start, inner_r=0.04568666666666668):
        cor_to_un, un_to_cor, boundary = self.pre_file(pre)
        nodes, nodes_r = self.get_nodes(msh)
        lines, elements, lines_r, elements_r = self.get_elements(msh)

        tmp = self.get_arrays(nodes, lines, elements, inner_r, un_to_cor, boundary, new_unknown_start)
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

    @staticmethod
    def compute_mesh_transfer(values, vtx, wts, dif, dif2, fill_value=np.nan):
        work = np.append(values, np.zeros(dif))
        ret = np.einsum('nj,nj->n', np.take(work, vtx), wts)
        ret[np.any(wts < 0, axis=1)] = fill_value
        ret = ret[:(np.size(ret) - dif2)]
        return ret

    @staticmethod
    def get_nodes(file):
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
            if row[1] != '0' and row[1] != '1' and row[1] != '2':
                node_dict[row[0]] = np.array([float(row[1]), float(row[2])])
                point_to_node[row[1] + ' ' + row[2]] = row[0]

        return node_dict, point_to_node

    @staticmethod
    def get_elements(file):
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

    @staticmethod
    def pre_file(file):
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

    @staticmethod
    def cart2pol(x, y):
        r = (x ** 2 + y ** 2) ** .5
        return r

    def get_arrays(self, nodes_dict, lines_dict, elements_dict, inner_r, unknown_to_cor, boundary, new_start=0):
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

        r = self.cart2pol(points_com[:, 0], points_com[:, 1])
        inner = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
        points_inner = points_com[inner]
        outer = np.where(abs(r) > abs(inner_r) - 1e-9)[0]
        points_outer = points_com[outer]

        r = self.cart2pol(points_bou[:, 0], points_bou[:, 1])
        inner_boundary_nodes = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
        points_bou_inner = points_bou[inner_boundary_nodes]
        outer_boundary_nodes = np.where(abs(r) > abs(inner_r) + 1e-7)[0]
        points_bou_outer = points_bou[outer_boundary_nodes]

        r = self.cart2pol(unknown_com[:, 0], unknown_com[:, 1])
        inner = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
        unknown_com_inner = unknown_com[inner]
        outer = np.where(abs(r) > abs(inner_r) + 1e-7)[0]
        unknown_com_outer = unknown_com[outer]

        r = self.cart2pol(unknown[:, 0], unknown[:, 1])
        inner = np.where(abs(r) - 1e-9 < abs(inner_r))[0]
        unknown_inner = unknown[inner]
        outer = np.where(abs(r) > abs(inner_r) + 1e-7)[0]
        unknown_outer = unknown[outer]

        r = self.cart2pol(unknown_new[:, 0], unknown_new[:, 1])
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

    def interpolation_factors(self, data_coarse, data_fine):

        # vtxCom, wtsCom = interp_weights(data_coarse['unknownCom'], data_fine['unknown'][len(data_coarse['unToCor']):])
        vtx_inner, wts_inner = self.interp_weights(data_coarse['unknownComInner'], data_fine['unknownNewInner'])
        vtx_outer, wts_outer = self.interp_weights(data_coarse['unknownComOuter'], data_fine['unknownNewOuter'])

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
