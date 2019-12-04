import numpy as np
import scipy.sparse as sp
import random
import os
import pickle

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# def preprocess_graph(adj):
#     adj = sp.coo_matrix(adj)
#     adj_ = adj + sp.eye(adj.shape[0])
#     rowsum = np.array(adj_.sum(1))
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     return sparse_to_tuple(adj_normalized)

def preprocess_graph(adj):
    # directed graph norm
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # D(-0.5)*A*D(-0.5)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # D(-0.5)*A
    # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().tocoo()
    # adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def hidden_edges(adj):
    '''
    每个节点随机的隐藏一个in 一个 out， 输出隐藏后的adj和隐藏的边
    :return:
    '''
    cache_path = './data/cora/cache.pkl'
    if os.path.isfile(cache_path):
        adj, h_in, h_out = pickle.load(open(cache_path, 'rb+'))
    else:
        dense_adj = np.array(adj.todense())
        h_out = []
        h_in = []
        for i in range(dense_adj.shape[0]):
            node_out = np.where(dense_adj[i, :] != 0)[0]
            node_in = np.where(dense_adj[:, i] != 0)[0]
            if len(node_out) != 0:
                r_node_out = random.choice(node_out)
                h_out.append((i, r_node_out))
            if len(node_in) != 0:
                r_node_in = random.choice(node_in)
                h_in.append((r_node_in, i))
        hidden_ = set(h_out).union(set(h_in))
        # 把选中的hidden边再adj中隐藏
        for e_i in hidden_:
            dense_adj[e_i] = 0
        k = 0
        # 统计没有out link的node
        for i in dense_adj:
            if i.max() == 0:
                k += 1
        adj = sp.csc_matrix(dense_adj)
        with open(cache_path, 'wb+') as f:
            pickle.dump((adj, h_in, h_out), f)
    return adj, h_in, h_out
# def hidden_edges(adj):
#     '''
#     每个节点随机的隐藏一个in 一个 out， 输出隐藏后的adj和隐藏的边
#     :return:
#     '''
#     dense_adj = np.array(adj.todense())
#     h_out = []
#     h_in = []
#     for i in range(dense_adj.shape[0]):
#         node_out = np.where(dense_adj[i, :] != 0)[0]
#         node_in = np.where(dense_adj[:, i] != 0)[0]
#         if len(node_out) != 0:
#             r_node_out = random.choice(node_out)
#             dense_adj[i, r_node_out] = 0
#             h_out.append([i, r_node_out])
#         if len(node_in) != 0:
#             r_node_in = random.choice(node_in)
#             dense_adj[r_node_in, i] = 0
#             h_in.append([r_node_in, i])
#     adj = sp.csc_matrix(dense_adj)
#     return adj, h_in, h_out
