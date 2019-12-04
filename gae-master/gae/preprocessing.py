import numpy as np
import scipy.sparse as sp
import random
import pickle
import os
import copy


def prediction_with_recall(pred, train_label, hidden_in, hidden_out, top_k=1):
    '''
    :param pred: [n,n] matrix, p(i,j) means (i->j)'s probability
    :param train_label: [n,n] matrix
    :param hidden_in:[m,2] list, each row means a edge from hidden_in[i,0] to hidden_in[i,1]
    :param hidden_out: [m,2] list, each row means a edge from hidden_out[i,0] to hidden_out[i,1]
    :param top_k: check top k scores to see if the link in the
    :return:
    hidden_in_recall: float, means the recall rate in hidden in  link
    hidden_out_recall: float, means the recall rate in hidden out link
    '''

    assert len(hidden_in) > 0
    assert len(hidden_out) > 0
    num_correct_in = np.zeros((top_k))

    for link_in in hidden_in:
        s_node = link_in[0]
        e_node = link_in[1]
        train_node = np.where(train_label[:, e_node] > 0)
        p = copy.deepcopy(pred[:, e_node])
        # exclude train link
        p[train_node[0]] = 0
        node_rank = np.argsort(p)[::-1]
        rank_k = np.where(node_rank == s_node)[0][0]
        if rank_k < top_k:
            num_correct_in[rank_k:] += 1

    num_correct_out = np.zeros((top_k))
    for link_out in hidden_out:
        s_node = link_out[0]
        e_node = link_out[1]
        train_node = np.where(train_label[s_node, :] > 0)
        p = copy.deepcopy(pred[s_node, :])
        # exclude train link
        p[train_node[0]] = 0
        node_rank = np.argsort(p)[::-1]
        rank_k = np.where(node_rank == e_node)[0][0]
        if rank_k < top_k:
            num_correct_out[rank_k:] += 1
    return num_correct_in/len(hidden_in), num_correct_out/len(hidden_out)


def hidden_edges(adj):
    '''
    每个节点随机的隐藏一个in 一个 out， 输出隐藏后的adj和隐藏的边
    :return:
    '''
    cache_path = './data/cache_citeseer.pkl'
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
        # 把选中的hidden边在adj中隐藏
        for e_i in hidden_:
            dense_adj[e_i] = 0
            dense_adj[e_i[::-1]] = 0
        k = 0
        # 统计没有out link的node
        for i in dense_adj:
            if i.max() == 0:
                k += 1
        adj = sp.csc_matrix(dense_adj)
        with open(cache_path, 'wb+') as f:
            pickle.dump((adj, h_in, h_out), f)
    return adj, h_in, h_out


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
