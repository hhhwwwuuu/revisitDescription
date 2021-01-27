import numpy as np
import pandas as pd
import networkx as nx
from . import buildGraph as bg
from numpy.linalg import eig, inv
from buildGraph import perList
from itertools import product
from scipy.stats import chi2_contingency

def normalization(matrix):
    matrix = np.array(matrix)
    _range = np.max(matrix) - np.min(matrix)
    return (matrix - np.min(matrix)) / _range


def filterRarePermission():
    data = pd.read_csv('../../data/permission_frequency_on_category.csv', encoding='utf-8')
    rarePermissions = {}
    permissions = data.columns.values.tolist()[1:11]
    for index, row in data.iterrows():
        for i in range(len(row[1:11])):
            # filiter out rare permissions that less than 5% ocurrence in dataset
            if row[i + 1] / row['Count'] < 0.05:
                if row['category'] in rarePermissions:
                    rarePermissions[row['category']].append(permissions[i])
                else:
                    rarePermissions[row['category']] = [permissions[i]]
    return rarePermissions


def freqPermission(category, permission):
    data = pd.read_csv('../../data/permission_frequency_on_category.csv', encoding='utf-8')
    return data.loc[data.category == category, permission].values[0] / \
           data.loc[data.category == category, 'Count'].values[0]


def weightProportion(src, dst, graph):
    weightMax = np.max(bg.weightMatrix(graph, mtype='weight'))
    return graph.edges[src, dst]['weight'] / weightMax

#
# def searchCommon(graph):
#     '''
#     :param graph:
#     :return: common permissions
#     '''
#
#     # generate the assoicated matrix
#     adj_matrix = bg.weightMatrix(graph, mtype='adjacency') + np.eye(len(bg.perList))
#     weight_matrix = bg.weightMatrix(graph, mtype='weight')
#     dg_matrix = bg.degreeMatrix(graph)
#
#     # check the optimal start node
#
#     # pick the nodes with large degree & high request frequency
#     # 获取度最大 且 请求次数最多的permissions集
#     row, _ = np.where(dg_matrix == np.max(dg_matrix))
#     maxDegree = np.array([graph.nodes[bg.perList[i]]['count'] for i in row])
#     roots = [bg.perList[i] for i in range(len(bg.perList)) if graph.nodes[bg.perList[i]]['count'] == np.max(maxDegree)]
#     # print(roots)
#
#     '''
#     从roots中的一个节点开始BFS
#     寻找common permissions的组合
#     '''
#     result = []
#     per = ['Calendar', 'Contacts', 'Camera', 'Location', 'Microphone', 'Phone', 'SMS', 'Call Log', 'Storage', 'Sensors']
#     for root in roots:
#         result.append(BFS(root, per, graph))
#     return result
#
#
# def DFS(root, permissions, graph):
#     common = []
#     common.append(root)
#
#     permissions.remove(root)
#
#     nodes = [u for u, _ in nx.bfs_predecessors(graph, source=root, depth_limit=1)]
#     # print(nodes)
#     neighbors = sortNeighbor(root, nodes, permissions, graph)
#     # print(neighbors)
#     '''
#     判断排序后的第一个是否满足要求，
#     若满足，选取该点作为common，
#     否则，结束
#     '''
#     if freqPermission(graph.graph['name'], neighbors[0][0]) < 0.05 or weightProportion(root, neighbors[0][0],
#                                                                                        graph) < 0.2:
#         return common
#     elif neighbors[0][0] in permissions:
#         common.extend(DFS(neighbors[0][0], permissions, graph))
#         return common
#     else:
#         return common
#
#
# def BFS(root, permissions, graph):
#     common = []
#
#     if len(permissions) == 0 or root not in permissions or root in common:
#         return common
#
#     common.append(root)
#     permissions.remove(root)
#
#     nodes = [u for u, _ in nx.bfs_predecessors(graph, source=root, depth_limit=1)]
#     neighbors = sortNeighbor(root, nodes, permissions, graph)
#
#     '''
#     广度优先筛选符合条件的permissions
#     若满足条件，选取为common
#     否则， 跳过
#     '''
#     # print(neighbors)
#     for key, values in neighbors:
#         if freqPermission(graph.graph['name'], key) < 0.05 or weightProportion(root, key, graph) < 0.2:
#             break
#         elif key in permissions:
#             common.append(key)
#             permissions.remove(key)
#
#     for node in common[1:]:
#         common.extend(BFS(node, permissions, graph))
#
#     return common
#
#
# # sort the neighbor by degree of node and weight of edges
# def sortNeighbor(root, nodes, permissions, graph):
#     results = {}
#     for node in nodes:
#         if node in permissions:
#             results[node] = {'degree': graph.degree[root],
#                              'weight': graph.edges[root, node]['weight'],
#                              'count': graph.nodes[node]['count']}
#
#     results = sorted(results.items(), key=lambda x: (x[1]['count'], x[1]['weight'], x[1]['degree']), reverse=True)
#     return results
#
#
# def spd(A):
#     eig_val, eig_vec = eig(A)
#     eig_diag = np.diag(1 / (eig_val ** 0.5))
#     B = np.dot(np.dot(eig_vec, eig_diag), inv(eig_vec))
#     return B
#
#
# def calcultate(graph):
#     degree = bg.degreeMatrix(graph)
#     adj = bg.weightMatrix(graph, mtype='adj')
#     print(degree)
#     print('--------------------adj matrix-------------------')
#     print(adj)
#     print('--------------------1/2 matrix-------------------')
#     a = np.mat(degree)
#     # print(inv(a))
#     print(degree * adj)
#     # print(np.power(degree, -1/2, where=(degree!=0))*adj*np.power(degree, -1/2, where=(degree!=0)))
#
#
# def compute_weight(graph, selected_nodes):
#     """
#     根据选择的权限，计算其总权重价值
#     1. 如果node中存在度为0的节点，该节点的权重直接给0
#     2. 针对每个节点，计算该节点在其他权限中的支持度。
#     3. 单个节点的情况下，以该权限的频度为返回值
#     4. 多个节点情况下， 返回平均支持度
#     :param graph:
#     :param selected_nodes:
#     :return:
#     """
#     if len(selected_nodes) < 1:
#         raise ValueError("The number of selected nodes should greater than or equal 1.")
#
#     if len(selected_nodes) == 1:
#         return freqPermission(graph.graph['name'], selected_nodes[0]) * gaussian(H(selected_nodes[0], graph, []))
#         # + (graph.nodes[selected_nodes[0]]['individual'] / graph.nodes[selected_nodes[0]]['count'])
#         # gaussian(H(selected_nodes[0], graph, []))
#         #       graph.nodes[selected_nodes[0]]['individual'] / graph.nodes[selected_nodes[0]]['count']
#         # return freqPermission(graph.graph['name'], selected_nodes[0]) *\
#         #       (graph.nodes[selected_nodes[0]]['individual'] / graph.nodes[selected_nodes[0]]['count'])
#     else:
#         # 对每一个permission的支持度进行计算并求和
#         result = []
#         for node in selected_nodes:
#             retained = list(set(selected_nodes) - set([node]))
#             # result.append(freqPermission(graph.graph['name'], node) * support(node, graph, retained))
#             result.append(freqPermission(graph.graph['name'], node) * gaussian(H(node, graph, retained)))
#             # gaussian(H(node, graph, tmp))
#         result = np.mean(result)
#         return result
#
#
# def gaussian(x):
#     return np.exp(-x ** 2)
#
#
# def support(src, graph, dst=[]):
#     if not dst:
#         return graph.nodes[src]['individual'] / graph.nodes[src]['count']
#     result = 0
#     for node in dst:
#         if graph.has_edge(src, node):
#             result += graph[src][node]['weight'] / graph.nodes[node]['count']
#         else:
#             result += 0
#     return result / len(dst)
#
#
# # compute information entropy
# def H(src, graph, dst=[]):
#     """
#     1. 如果边不存在， 直接给 1， 因为没有支持度
#     2. 如果边存在 且 概率为1， 直接赋0， 支持度最高
#     :param src:
#     :param graph:
#     :param dst:
#     :return:
#     """
#     if not dst:
#         if graph.nodes[src]['individual'] == graph.nodes[src]['count']:
#             return 0
#         elif graph.nodes[src]['individual'] == 0:
#             return 1
#         else:
#             ratio = graph.nodes[src]['individual'] / graph.nodes[src]['count']
#             return -ratio * np.log2(ratio)
#     result = 0
#     for node in dst:
#         if graph.has_edge(src, node):
#             ratio = graph[src][node]['weight'] / graph.nodes[node]['count']
#             result += -1 * ratio * np.log2(ratio) \
#                 if graph[src][node]['weight'] != graph.nodes[node]['count'] else 0
#         else:
#             result += 1
#     return result
#
#
# def permission_selection(graph):
#     count = 0
#     retainList = []
#     for per in perList:
#         if graph.nodes[per]['count'] != 0:
#             count += 1
#             retainList.append(per)
#     selected_permissions = list(product(range(2), repeat=count))
#
#     # print(selected_permissions)
#     # 根据全排列 生成对应的permissions list
#     selected_results = []
#     for comb in selected_permissions:
#         selected_results.append([x for x, y in zip(retainList, list(comb)) if y == 1])
#     if [] in selected_results:
#         selected_results.remove([])
#     return selected_results
#
#
# def find_commons(graph):
#     selected_permissions = permission_selection(graph)
#
#     weight = {compute_weight(graph, permissions): permissions for permissions in selected_permissions}
#
#     # 结果按value排序，取最大weight作为结果
#     result = sorted(weight.items(), key=lambda x: [x[0], x[1]], reverse=True)
#     return result


def compute_chi(total, a_count, b_count, weight):
    x = np.array([[weight, a_count-weight],[b_count-weight, total-a_count-b_count+weight]])
    return 1 if chi2_contingency(x, correction=False)[1] < 0.05 else 0

def count_unused_permissions(graph):
    num = 0
    for per in perList:
        if graph.nodes[per]['count'] == 0:
            num += 1
    return num

def coverage_permission(permission, graph):
    return graph.nodes[permission]['count'] / graph.graph['num']

def find_common_chi(data, graph):
    num_permissions = [sum(list(row[2:])) for index, row in data.iterrows()]
    selected_num = np.median(num_permissions)
    #print("The median of {} requested permissions is {}".format(graph.graph['name'], selected_num))

    per_score = {}
    null_num = count_unused_permissions(graph)

    for per in perList:
        sum_chi = 0
        # 如果当前permission被任意app请求过，计算其与其他相关性。
        # 否则，该permission相关性总和为0
        if graph.nodes[per]['count'] != 0:
            for cmp_per in perList:
                if per == cmp_per or graph.nodes[cmp_per]['count'] == 0 or not graph.has_edge(per, cmp_per):
                    # 自身不计算
                    # 被比较权限未被调用不计算
                    # 两个permission未在任何app中共存过 不计算
                    continue
                else:
                    sum_chi += compute_chi(total=len(data), a_count=graph.nodes[per]['count'],
                                           b_count=graph.nodes[cmp_per]['count'],
                                           weight=graph.edges[per,cmp_per]['weight'])
        cor_score = sum_chi / (len(perList) - null_num - 1)
        score = coverage_permission(per, graph) * cor_score
        per_score[per] = score

    result = sorted(per_score.items(), key = lambda x: [x[1], x[0]], reverse=True)


    # 提取common permissions
    # 如果提取的最后一个的value和 下一个完全一致，超额提取median+1个permissions
    com_per = [result[i][0] for i in range(int(selected_num))]
    if result[int(selected_num)-1][1] == result[int(selected_num)][1]:
        com_per.append(result[int(selected_num)][0])

    return com_per, selected_num


