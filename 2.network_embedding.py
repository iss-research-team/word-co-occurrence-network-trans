#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 下午3:56
# @Author  : liu yuhan
# @FileName: 2.network_embedding.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import sys
import os
from tqdm import tqdm


def not_empty(s):
    return s and s.strip()


class NetworkDeal():
    '''
    生成网络，
    '''

    def __init__(self, node_path, link_path, ng_num):
        self.ng_num = ng_num
        self.node_list = open(node_path, 'r', encoding='UTF-8').readlines()
        self.node_num = len(self.node_list)
        with open(link_path, 'r', encoding='UTF-8') as file:
            self.link_list_weighted = json.load(file)
        print('node_num:', self.node_num)
        print('link_num:', len(self.link_list_weighted))

    def get_network_feature(self):
        '''
        构建网络计算个重要参数——degree
        :return:
        '''
        g = nx.Graph()
        g.add_nodes_from([i for i in range(self.node_num)])
        g.add_weighted_edges_from(self.link_list_weighted)
        self.node_degree_dict = dict(nx.degree(g))

    def get_data(self):
        '''
        负采样
        :return:
        '''
        # 负采样
        sample_table = []
        sample_table_size = 1e8
        # 词频0.75次方
        pow_frequency = np.array(list(self.node_degree_dict.values())) ** 0.75
        nodes_pow = sum(pow_frequency)
        ratio = pow_frequency / nodes_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            sample_table += [wid] * int(c)
        sample_table = np.array(sample_table)

        # 生成一个训练集：
        source_list = []
        target_list = []
        node_ng_list = []

        for link in self.link_list_weighted:
            source_list.append(link[0])
            target_list.append(link[1])
            node_ng = np.random.choice(sample_table, size=(self.ng_num)).tolist()
            node_ng_list.append(node_ng)

        return self.node_num, \
               torch.LongTensor(source_list), torch.LongTensor(target_list), torch.LongTensor(node_ng_list)


class Line(nn.Module):
    def __init__(self, word_size, dim):
        super(Line, self).__init__()
        initrange = 0.5 / dim
        # input
        print('weight init...')
        self.u_emd = nn.Embedding(word_size, dim)
        self.u_emd.weight.data.uniform_(-initrange, initrange)
        self.context_emd = nn.Embedding(word_size, dim)
        self.context_emd.weight.data.uniform_(-0, 0)

    # 这边进行一个一阶+二阶的
    def forward(self, s, t, ng):
        vector_i = self.u_emd(s)
        # 一阶
        vector_o1 = self.u_emd(t)
        vector_ng1 = self.u_emd(ng)
        output_1_1 = torch.matmul(vector_i, vector_o1.transpose(-1, -2)).squeeze()
        output_1_1 = F.logsigmoid(output_1_1)
        # 负采样的部分
        output_1_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng1.transpose(-1, -2)).squeeze()
        output_1_2 = F.logsigmoid(-1 * output_1_2).sum(1)

        output_1 = -1 * (output_1_1 + output_1_2)
        # 二阶
        vector_o2 = self.context_emd(t)
        vector_ng2 = self.context_emd(ng)
        output_2_1 = torch.matmul(vector_i, vector_o2.transpose(-1, -2)).squeeze()
        output_2_1 = F.logsigmoid(output_2_1)
        # 负采样的部分
        output_2_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng2.transpose(-1, -2)).squeeze()
        output_2_2 = F.logsigmoid(-1 * output_2_2).sum(1)

        # 组合
        output_2 = -1 * (output_2_1 + output_2_2)

        loss = torch.mean(output_1) + torch.mean(output_2)

        return loss

    # 保存参数
    def save_embedding(self, file_name):
        """
        Save all embeddings to file.
        """
        embedding = self.u_emd.weight.cpu().data.numpy()
        np.save(file_name, embedding)


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, s_list, t_list, ng_list):
        self.s_list = s_list
        self.t_list = t_list

        self.ng_list = ng_list

    def __len__(self):
        return len(self.s_list)

    def __getitem__(self, idx):
        return self.s_list[idx], self.t_list[idx], self.ng_list[idx]


def train():
    # 参数设置
    d = 768
    ng_num = 5
    batch_size = 128
    epochs = 50
    cuda_order = '0'
    if_trans = 'yes'

    # 数据载入
    print('processing---')
    node_path = "data/input/keyword.txt"

    if if_trans == 'yes':
        link_path = "data/output/link-transfer.json"
        node_emb_path = "data/output/node_embed_transfer"
    else:
        link_path = "data/output/link-origin.json"
        node_emb_path = "data/output/node_embed_origin"

    # 数据处理
    networkdeal = NetworkDeal(node_path, link_path, ng_num)
    networkdeal.get_network_feature()
    node_size, s_list, t_list, ng_list = networkdeal.get_data()
    loader = Data.DataLoader(MyDataSet(s_list, t_list, ng_list), batch_size, True)

    # cuda
    device = torch.device("cuda:" + cuda_order if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # 模型初始化
    line = Line(node_size, d)
    line.to(device)
    optimizer = optim.Adam(line.parameters(), lr=0.0001)

    # 保存平均的loss
    loss_min = 100000
    loss_min_epoch = 0

    with tqdm(total=epochs) as bar:
        for epoch in range(epochs):
            loss_collector = []
            for i, (s, t, ng) in enumerate(loader):
                s = s.to(device)
                t = t.to(device)
                ng = ng.to(device)
                loss = line(s, t, ng)
                loss.backward()
                optimizer.step()
                loss_collector.append(loss.item())
            ave_loss = np.mean(loss_collector)
            bar.set_description('Epoch ' + str(epoch))
            bar.set_postfix(loss=ave_loss)
            bar.update(1)
            if ave_loss < loss_min:
                line.save_embedding(node_emb_path)
                loss_min = ave_loss
                loss_min_epoch = epoch
    print('---min loss: %0.4f' % loss_min + ' ---min loss epoch: %d' % loss_min_epoch)


if __name__ == '__main__':
    train()
