import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import numpy as np


def load_data(dataset):
    files = ['cites', 'content']
    # 将id映射到所在的index 为了调整feature matrix的行的顺序
    if dataset == 'citeseer':
        dataset_name = 'citeseer_filter'
    else:
        dataset_name = dataset

    # 读有向边形成邻接矩阵
    graph = nx.read_edgelist('data/{}/{}.{}'.format(dataset, dataset_name, files[0]),
                             create_using=nx.DiGraph)
    adj = nx.adjacency_matrix(graph)
    # den_adj = np.array(adj.todense())

    with open('data/{}/{}.{}'.format(dataset, dataset_name, files[0]), 'rb') as f:
        node_index = {}
        index = 0
        for line in f.readlines():
            line_ = line.strip(b'\n \r').split(b'\t')
            if line_[0].decode() not in node_index.keys():
                node_index[line_[0].decode()] = index
                index += 1
            if line_[1].decode() not in node_index.keys():
                node_index[line_[1].decode()] = index
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
            feature.append([i.decode() for i in line_])
        df_feature = pd.DataFrame(feature)
        df_feature = df_feature.rename(columns={0: 'id'})
    df = pd.merge(df_node_index, df_feature, how='left', on='id')
    df_feature_reindex = df.iloc[:, 2:].astype(int)
    feature = sp.lil_matrix(df_feature_reindex.values)
    return adj.T, feature


def process_citeseer():
    # citeseer 数据集中边数据集有3327个节点，但是只有3312个节点有特征
    # 解决方法，把边集合中的多余节点删除（即删除包含多余节点的边）
    with open('data/citeseer/citeseer.content', 'rb') as f:
        paper_name_list = []
        for line in f.readlines():
            paper_name = line.strip(b'\n').split(b'\t')[0]
            paper_name_list.append(paper_name)
    with open('data/citeseer/citeseer.cites', 'rb') as f:
        with open('data/citeseer/citeseer_filter.cites', 'w') as ff:
            for line in f.readlines():
                line_ = line.strip(b'\n').split(b'\t')
                if line_[0] not in paper_name_list or line_[1] not in paper_name_list:
                    continue
                elif line_[0] == line_[1]:
                    continue
                else:
                    ff.write(line.decode())
    print('done!')


# def check_data():
#     graph1 = nx.read_edgelist('data/citeseer/citeseer.cites',
#                              create_using=nx.DiGraph)
#     graph2 = nx.read_edgelist('data/citeseer/citeseer_filter.cites',
#                              create_using=nx.DiGraph)
#     node_1 = set(graph1.node)
#     node_2 = set(graph2.node)
#     a = node_2.difference(node_1)
#     b = node_1.difference(node_2)
#     print(1)


if __name__ == "__main__":
    load_data('citeseer')  # citeseer cora
    # process_citeseer()
    # check_data()