#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:yuyi
# datetime:2019/9/23 13:49
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
import copy

def fast_test(pred, train_label, hidden_in, hidden_out, top_k=1):
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

    num_node = train_label.shape[0]
    # 将在train label中的边设为0
    pred[np.where(train_label > 0)] = 0
    # 处理hidden out link
    out_have_index = np.array(hidden_out)[:, 0]
    # 删除没有 hidden out的节点
    hidden_out_label = np.array(hidden_out)[:, 1].reshape(-1, 1)
    pred_out = pred[out_have_index]

    # 升序排列
    out_argsort = np.argsort(-pred_out, axis=1)
    out_rank = np.where(out_argsort == hidden_out_label)[1]
    recall_out = np.zeros((top_k))
    for i in out_rank:
        if i < top_k:
            recall_out[i:] += 1

    # 处理hidden in link
    in_have_index = np.array(hidden_in)[:, 1]
    # in_delete_index = set(range(num_node)) - in_have_index
    # 删除没有hidden in 的节点
    hidden_in_label = np.array(hidden_in)[:, 0].reshape(1, -1)
    pred_in = pred[:, in_have_index]
    # pred_in = np.delete(pred, list(in_delete_index), axis=1)
    # 升序排列
    in_argsort = np.argsort(-pred_in, axis=0)
    in_rank = np.where(in_argsort == hidden_in_label)[0]
    recall_in = np.zeros((top_k))
    for i in in_rank:
        if i <top_k:
            recall_in[i:] += 1
    return recall_in/in_rank.shape[0], recall_out/out_rank.shape[0]


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


def plot_lines_fig(df,
                   x_index=0,
                   x_name='step',
                   line_index=[1],
                   line_name=['气温'],
                   line_range=[26, 36],
                   line_y_name='recall',
                   if_loss=True,
                   table_title='results',
                   loss_index=1,
                   loss_range=[0, 9]
                   ):

    fig = plt.figure(figsize=(30, 8))
    ax1 = fig.add_subplot(111)
    num_day = df.shape[0]
    # 画 line

    ax1.set_ylim(line_range[0], line_range[1])
    for index, name in zip(line_index, line_name):
        ax1.plot(range(num_day), df.iloc[:, index].astype(float), label=name)
    index = [float(c) + 0.4 for c in range(num_day)]
    plt.xticks(index, df.iloc[:, x_index])
    for label in ax1.get_xticklabels()[::2]:
        label.set_visible(False)
    plt.xticks(rotation=-90)
    # 画 wether
    if if_loss:
        ax2 = ax1.twinx()
        ax2.set_ylim(loss_range[0], loss_range[1])
        ax2.plot(range(num_day), df.iloc[:, loss_index].astype(float), 'g*-', label='loss')
    # 显示坐标轴信息
    ax1.set_xlabel(x_name)
    plt.title(table_title)
    ax1.set_ylabel(line_y_name)
    if if_loss:
        ax2.set_ylabel('loss')
        ax2.legend(loc='upper right')
    ax1.legend(loc='upper left')

    plt.savefig('./results/' + table_title+'.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    # plt.show()


def set_configs(**kwargs):
    configs = list()
    base_config = dict()
    base_config['learning_rate'] = 0.005
    base_config['epochs'] = 2000
    # base_config['epochs'] = 60
    base_config['hidden1'] = 32
    base_config['hidden2'] = 16
    base_config['weight_decay'] = 0.
    base_config['dataset'] = 'cora'
    base_config['if_BN'] = True
    base_config['features'] = 1
    base_config['weight_init'] = 'glorot'   # 'truncated_normal'
    base_config['result_name'] = ''
    base_config['result_path'] = './results/' + base_config['result_name'] + '.csv'
    allowed_kwargs = {'learning_rate', 'weight_decay', 'if_BN'}
    for kwarg in kwargs.keys():
        assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
    parameter_list = list(itertools.product(*kwargs.values()))
    for p_set in parameter_list:
        result_name = ''
        for i, key in enumerate(kwargs.keys()):
            base_config[key] = p_set[i]
            result_name += '_' + key + '_%s' % str(p_set[i])
        base_config['result_name'] = result_name
        base_config['result_path'] = './results/' + base_config['result_name'] + '.csv'
        configs.append(copy.deepcopy(base_config))
    return configs


def plot_all(result_path, title_name):
    df = pd.read_csv(result_path)
    plot_lines_fig(df,
                   x_index=0,
                   x_name='step',
                   line_index=list(range(1, 21)),
                   line_name=df.columns[1:21],
                   line_range=[0, 0.1],
                   if_loss=True,
                   loss_index=-1,
                   loss_range=[0, 8],
                   table_title='in_link_recall_loss_fig_' + title_name
                   )
    plot_lines_fig(df,
                   x_index=0,
                   x_name='step',
                   line_index=list(range(21, 41)),
                   line_name=df.columns[21:41],
                   line_range=[0, 0.3],
                   if_loss=True,
                   loss_index=-1,
                   loss_range=[0, 8],
                   table_title='out_link_recall_loss_fig_' + title_name
                   )


if __name__ == '__main__':
    # df = pd.read_csv('./results/base_line_lr_0.005_with_weight_init_glorot_weight_decay_5e-4__6.csv')
    # plot_lines_fig(df,
    #                x_index=0,
    #                x_name='step',
    #                line_index=list(range(1, 21)),
    #                line_name=df.columns[1:21],
    #                line_range=[0, 0.1],
    #                if_loss=True,
    #                loss_index=-1,
    #                loss_range=[0, 8],
    #                table_title='in link recall loss fig with glorot weight decay 5e-4 retry6'
    #                )
    # plot_lines_fig(df,
    #                x_index=0,
    #                x_name='step',
    #                line_index=list(range(21, 41)),
    #                line_name=df.columns[21:41],
    #                line_range=[0, 0.3],
    #                if_loss=True,
    #                loss_index=-1,
    #                loss_range=[0, 8],
    #                table_title='out link recall loss fig with glorot weight decay 5e-4 retry6'
    #                )
    configs = set_configs(weight_decay=[1e-4, 5e-4], weight_init=['truncated_normal', 'glorot'])
    print(1)
