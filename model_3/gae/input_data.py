import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import numpy as np

# def parse_index_file(filename):
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index
#
#
# def load_data(dataset):
#     # load the data: x, tx, allx, graph
#     names = ['x', 'tx', 'allx', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#     x, tx, allx, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
#     test_idx_range = np.sort(test_idx_reorder)
#
#     if dataset == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#
#     return adj, features


def load_data(dataset):
    files = ['cites', 'content']

    # 读有向边形成邻接矩阵
    graph = nx.read_edgelist('data/{}/{}.{}'.format(dataset, dataset, files[0]),
                             create_using=nx.DiGraph)
    adj = nx.adjacency_matrix(graph)
    # den_adj = np.array(adj.todense())

    # 将id映射到所在的index 为了调整feature matrix的行的顺序
    with open('data/{}/{}.{}'.format(dataset, dataset, files[0]), 'rb') as f:
        node_index = {}
        index = 0
        for line in f.readlines():
            line_ = line.strip(b'\n').split(b'\t')
            if int(line_[0]) not in node_index.keys():
                node_index[int(line_[0])] = index
                index += 1
            if int(line_[1]) not in node_index.keys():
                node_index[int(line_[1])] = index
                index += 1
    node_index_list = []
    for key, value in node_index.items():
        node_index_list.append([key, value])
    df_node_index = pd.DataFrame(node_index_list, columns=['id', 'index'])

    # id_to_index = {}
    # index_to_id = {}
    # i = 0
    # for node in graph.nodes:
    #     id_to_index[int(node)] = i
    #     index_to_id[i] = int(node)
    #     i += 1

    # 读边的信息，调整行的顺序，转为稀疏矩阵。
    with open('data/{}/{}.{}'.format(dataset, dataset, files[1]), 'rb') as f:
        feature = []
        for line in f.readlines():
            line_ = line.strip(b'\n').split(b'\t')[:-1]
            feature.append(line_)
        df_feature = pd.DataFrame(feature).astype(int)
        df_feature = df_feature.rename(columns={0: 'id'})
    df = pd.merge(df_node_index, df_feature, how='left', on='id')
    df_feature_reindex = df.drop(['id', 'index'], axis=1)
    feature = sp.lil_matrix(df_feature_reindex.values)
    return adj.T, feature


if __name__ == "__main__":
    load_data('cora')
