from itertools import combinations
import networkx as nx
import pandas as pd
import numpy as np
import os

perList = ['Calendar', 'Contacts', 'Camera', 'Location', 'Microphone', 'Phone', 'SMS', 'Call Log', 'Storage', 'Sensors']

def dataReader():
    print(os.getcwd())
    data = pd.read_csv('../data/permission_preprocessed.csv', encoding= 'utf-8')
       

    def getPermissions(data):
        perList = data.columns.values.tolist()[2:]
        return perList

    def getCategory(data):
        return list(set(data['category'].values.tolist()))

    return getCategory(data), getPermissions(data), data


def buildGraph(category, permissions, data):
    '''
    generate the graph of permission relation based on existing data for each genre
    :param category:
    :param permissions:
    :param data:
    :return:
    '''
    ### connecting the relations of permissions based on apps
    def connectEdge(graph):
        for index, row in data.iterrows():
            if 1 in row[2:].values.tolist():
                reqPermissions = [permissions[i] for i in range(len(permissions)) if row[2:].values.tolist()[i]]
                #print(reqPermissions)
                if len(reqPermissions) != 1:
                    edges = combinations(reqPermissions, 2)
                    for edge in edges:
                        if graph.has_edge(edge[0], edge[1]):
                            graph[edge[0]][edge[1]]['weight'] += 1
                        else:
                            graph.add_edge(edge[0], edge[1], weight=1)
            else:
                continue
        return graph

    # create an empyt graph, namly category of apps
    graph = nx.Graph(name=category)

    # init all nodes by permissions
    graph.add_nodes_from(permissions)

    '''
    add attributes to each node
    name : naming the node with associated permission
    count: count the requested times of each permissions
    '''
    for per in permissions:
        graph.nodes[per]['name'] = per
        try:
            graph.nodes[per]['count'] = data[per].value_counts()[1]
        except:
            graph.nodes[per]['count'] = 0

    graph = connectEdge(graph)

    return graph

def weightMatrix(graph, mtype = 'weight'):
    '''
    generate the weight matrix or adjacency matrix for the graph
    :param graph: the graph
    :param mtype: matrix type (only 'weight' and 'adjacency'). Default is weight matrix
    :return: generated matrix
    '''

    def adjacencyNodes(graph):
        adjList = [[node, n_neigh] for node, n_neigh in graph.adjacency()]
        return adjList

    adjList = adjacencyNodes(graph)

    adj_matrix = np.zeros(shape=[len(perList), len(perList)]).astype(np.int32)

    for node, neighbors in adjList:
        row = perList.index(node)
        for key, val in neighbors.items():
            adj_matrix[row, perList.index(key)] = val['weight'] if mtype == 'weight' else 1
    #print(adj_matrix)
    return np.array(adj_matrix)

def degreeMatrix(graph):

    degree_matrix = np.zeros(shape=[len(perList), len(perList)]).astype(np.int32)
    for index in range(len(perList)):
        degree_matrix[index, index] = graph.degree[perList[index]]
    #print(degree_matrix)
    return np.array(degree_matrix)