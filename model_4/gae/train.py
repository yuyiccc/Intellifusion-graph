from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from gae.model import GCNModelAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, hidden_edges
from gae.util import prediction_with_recall, fast_test


def train_and_test(config):
    tf.reset_default_graph()
    # Load data
    adj, features = load_data(config['dataset'])

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj, h_in, h_out = hidden_edges(adj)
    # 测试hidden edges输出的隐藏边的条数不同
    # for i in range(10):
    #     adj_train, h_in, h_out = hidden_edges(adj)
    #     print(adj_train.size, len(h_in), len(h_out))

    if config['features'] == 0:
        features = sp.identity(features.shape[0])  # featureless

    # # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'weight_decay': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]

    # Create model
    model = GCNModelAE(placeholders, num_nodes, config=config, num_features=num_features)

    # pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    opt = OptimizerAE(preds=model.reconstructions,
                      labels=tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                       validate_indices=False),
                      lr=config['learning_rate'])

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj
    adj_label = sparse_to_tuple(adj_label)
    result_data = []
    top_k = 20
    # Train model
    best_mean_out_recall = 0
    best_out_recall = None
    best_in_recall = None
    best_step = 0
    for epoch in range(config['epochs']):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['weight_decay']: config['weight_decay']})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.loss, opt.preds, opt.labels, opt.weight_decay_losses], feed_dict=feed_dict)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "time=", "{:.5f}".format(time.time() - t))
        if (epoch+1) % 10 == 0:
            # recall_in, recall_out = fast_test(outs[2], outs[3], hidden_in=h_in, hidden_out=h_out, top_k=20)
            recall_in, recall_out = prediction_with_recall(outs[2],
                                                           outs[3],
                                                           hidden_in=h_in,
                                                           hidden_out=h_out,
                                                           top_k=top_k)
            if np.mean(recall_out) > best_mean_out_recall:
                best_out_recall = recall_out
                best_in_recall = recall_in
                best_step = epoch
            result_data.append([epoch+1]+list(np.concatenate((recall_in, recall_out)))+[outs[1]])
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "test_recall_in=", "{:.5f}".format(recall_in[0]),
                  "test_recall_out=", "{:.5f}".format(recall_out[0]),
                  "time=", "{:.5f}".format(time.time() - t))
        if epoch == (config['epochs']+10):  # +10表示不用，-1表示在最后输出结果图
            popularity = sess.run(model.ck, feed_dict=feed_dict)
            plt.hist(popularity[0], bins=30)
            # plt.xticks(range(min(popularity[0][0]), max(popularity[0][0]))[::2], fontsize=8)
            plt.grid(linestyle="--", alpha=0.5)
            plt.xlabel('p value')
            plt.ylabel("number of nodes")
            plt.title('p values histogram')
            plt.savefig('./results/popularity_fig.png')
            plt.show()
        # test loss in optimizer
        # if epoch == 2198:
        #     out = sess.run([opt.preds,
        #                     opt.labels,
        #                     opt.positive_pred,
        #                     opt.location,
        #                     opt.grads_vars,
        #                     model.z_mean,
        #                     opt.loss], feed_dict=feed_dict)
        #     print(1)
        #
        #     print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        #           "train_acc=", "{:.5f}".format(outs[2]),
        #           "time=", "{:.5f}".format(time.time() - t))

    pd.DataFrame(result_data,
                 columns=['step'] +
                         ['recall_in@%d' % (i+1) for i in range(top_k)] +
                         ['recall_out@%d' % (i+1) for i in range(top_k)] +
                         ['train_loss']).\
        to_csv(config['result_path'], index=False)
    print("Optimization Finished!")
    return [config['result_name']]+list(best_in_recall)+list(best_out_recall)


if __name__ == '__main__':
    config = {}
    config['learning_rate'] = 0.0005
    config['epochs'] = 2000
    config['hidden1'] = 32
    config['hidden2'] = 16
    config['weight_decay'] = 0
    config['dataset'] = 'citeseer'  # 'cora'
    config['features'] = 1
    config['if_BN'] = True
    config['weight_init'] = 'glorot'
    config['result_name'] = 'test_citeseer.csv'
    config['result_path'] = os.path.join('./results', config['result_name'])
    train_and_test(config)
