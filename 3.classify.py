#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 下午3:56
# @Author  : liu yuhan
# @FileName: 2.network_embedding.py
# @Software: PyCharm
import json
import networkx as nx

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_data(if_trans):
    """
    针对字典会自动把int形key转换成str的情况
    我只能说 pickle yyds
    :return:
    """

    if if_trans == 'yes':
        x_path = 'data/output/node_embed_transfer.npy'
        link_path = 'data/output/link-transfer.json'
    else:
        x_path = 'data/output/node_embed_origin.npy'
        link_path = 'data/output/link-origin.json'

    x = np.load(x_path).tolist()
    with open('data-for-test/label_list.json', 'r', encoding='UTF-8') as file:
        y = json.load(file)
    with open(link_path, 'r', encoding='UTF-8') as file:
        link_list_weighted = json.load(file)
    g = nx.Graph()
    g.add_weighted_edges_from(link_list_weighted)
    node_list = list(g.nodes())
    x_choose = []
    y_choose = []
    for i in range(len(y)):
        if i in node_list:
            x_choose.append(x[i])
            y_choose.append(y[i])

    return torch.Tensor(x_choose), torch.LongTensor(y_choose)


# loss曲线绘制
def loss_draw(epochs, loss_list):
    plt.plot([i + 1 for i in range(epochs)], loss_list)
    plt.show()


class Dnn(nn.Module):
    def __init__(self, dim, hid_1, d_output):
        super(Dnn, self).__init__()
        self.liner_1 = nn.Sequential(nn.Linear(dim, hid_1), nn.BatchNorm1d(hid_1), nn.ReLU(True))

        self.liner_list = nn.Sequential()
        for i in range(5):
            self.liner_list.add_module('liner-layers-{}'.format(i), nn.Linear(hid_1, hid_1))
            self.liner_list.add_module('BN-layers-{}'.format(i), nn.BatchNorm1d(hid_1))
            self.liner_list.add_module('Relu-layers-{}'.format(i), nn.ReLU(True))

        self.liner_2 = nn.Sequential(nn.Linear(hid_1, dim), nn.BatchNorm1d(dim), nn.ReLU(True))
        self.liner_3 = nn.Linear(dim, d_output)

    def forward(self, x):
        output = self.liner_1(x)
        output = self.liner_list(output)
        output = self.liner_2(output)
        output = self.liner_3(output)
        return output


class MyDataSet(Data.Dataset):
    """
    没啥说的，正常的数据载入
    """

    def __init__(self, x, label):
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]


class Evaluator:
    def __init__(self, if_trans):
        self.f1_best = 0
        self.status_best = []
        if if_trans == 'yes':
            self.model_save_path = 'data-for-test/dnn_trans.pt'
        else:
            self.model_save_path = 'data-for-test/dnn_origin.pt'

    def evaluate(self, epoch, model, test_x, test_y, loss):
        torch.set_grad_enabled(False)
        out = model(test_x.cuda())
        prediction = torch.max(out.cpu(), 1)[1]
        pred_y = prediction.data.numpy()
        target_y = test_y.data.numpy()

        precision = precision_score(target_y, pred_y, average='weighted', zero_division=1)
        recall = recall_score(target_y, pred_y, average='weighted', zero_division=1)
        f1 = f1_score(target_y, pred_y, average='weighted', zero_division=1)
        status = ["epoch", epoch, "loss", loss, 'precision:', precision, 'recall:', recall, 'f1:', f1]
        print(status)
        if self.f1_best < f1:
            self.f1_best = f1
            self.status_best = status
            torch.save(model.state_dict(), self.model_save_path)

        torch.set_grad_enabled(True)


if __name__ == '__main__':
    batch_size = 16
    epochs = 100
    # 载入数据
    if_trans = 'no'
    data_x, data_y = get_data(if_trans)

    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, train_size=0.7)
    loader = Data.DataLoader(MyDataSet(data_x_train, data_y_train), batch_size, True)
    # 参数设置
    dim = 768
    h_1 = dim * 2
    d_o = 15
    # 模型初始化
    dnn = Dnn(dim, h_1, d_o)
    dnn.cuda()
    fun_loss = nn.CrossEntropyLoss()
    fun_loss.cuda()
    optimizer = optim.Adam(dnn.parameters(), lr=0.00001)

    # 计算平均的loss
    ave_loss = []

    # 保存
    evaluator = Evaluator(if_trans)

    for epoch in tqdm(range(epochs)):
        loss_collector = []
        for i, (x, label) in enumerate(loader):
            optimizer.zero_grad()
            x = x.cuda()
            label = label.cuda()
            output = dnn(x)
            loss = fun_loss(output, label)
            loss.backward()
            optimizer.step()

            loss_collector.append(loss.item())
        ave_loss.append(np.mean(loss_collector))
        evaluator.evaluate(epoch, dnn, data_x_test, data_y_test, np.mean(loss_collector))

    print(evaluator.status_best)
    loss_draw(epochs, ave_loss)
