# -*- coding: utf-8 -*-
import time
import os
import torch.nn as nn
import numpy as np
from collections import Counter
import pandas as pd
import torch
import matplotlib.pyplot as plt
from utils.metric import Confusion
from torch.utils.data import DataLoader
from utils.preprocess import *
from my_models.UPCD import UPCD
from datetime import datetime
import argparse
from make_dataset.Datasets import Datasets
from torch.optim import lr_scheduler
import warnings
import logging

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.ERROR)
# 忽略特定警告类
warnings.filterwarnings("ignore", message="You are resizing the embedding layer without providing a pad_to_multiple_of parameter.*")
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def train_epoch(model, epoch, train_loader, optimizer):
    t = time.time()
    total_loss = []
    model.train()
    flag = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = [item.float().to(model.device) for item in [data, label]]
        if flag == 0 and epoch == 0:
            print('data shape:', data.shape)
        loss, clusters = model(data)


        optimizer.zero_grad()
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        flag = 1

    print('Epoch: {:04d}'.format(epoch + 1),
          'total_loss: {:.4f}'.format(np.mean(total_loss)),
          'time: {:.4f}s'.format(time.time() - t))


def test_epoch(model, data_loader, e):
    model.eval()
    all_labels = []
    all_clusters = []
    for x, label in data_loader:
        x, label = [item.float().to(model.device) for item in [x, label]]
        with torch.no_grad():
            loss, clusters = model(x)

        if len(all_labels) <= 0:
            all_labels = label
            all_clusters = clusters
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_clusters = torch.cat((all_clusters, clusters), dim=0)


    print('all_clusters:', all_clusters.shape)
    print('all_labels:', all_labels.shape)

    all_labels = all_labels.cpu().detach().numpy()
    all_clusters = all_clusters.cpu().detach().numpy()

    # unique_values, counts = np.unique(all_clusters, return_counts=True)
    # for i in range(len(unique_values)):
    #     print(unique_values[i], counts[i])
    # np.save('./tcps/cluster_result_'+ str(e) + '.npy', all_clusters)
    # arr = np.load('array_file.npy')
    unique_values = np.unique(all_labels)
    cluster_num = len(unique_values)
    print("cluster_num:", cluster_num)

    cluster_indices = find_indices(all_clusters.reshape((all_clusters.shape[0])))
    predicted_labels = np.zeros_like(all_labels)
    for key, value in cluster_indices.items():
        print(f"clusters: {key}")
        labels = all_labels[value]
        unique_values, counts = np.unique(labels, return_counts=True)
        for i in range(len(unique_values)):
            print(unique_values[i], counts[i])
        counter = Counter(labels)
        most_common_element, count = counter.most_common(1)[0]
        # print(f'The element that appears most often is {most_common_element}, which appears {count} times.')
        predicted_labels[value] = most_common_element

    predicted_labels = torch.tensor(predicted_labels)
    all_labels = torch.tensor(all_labels)

    confusion = Confusion(cluster_num)
    confusion.add(predicted_labels, all_labels)
    confusion.optimal_assignment(args.cluster_space)

    acc = confusion.acc()
    cluster_score = confusion.clusterscores()
    print("epoch:{}, cluster_score:{}, ACC:{}".format(e, cluster_score, acc))

    return acc, cluster_score




def solver(args, model, train_loader):
    best_acc = 0
    best_cluster_score = {}
    best_epoch = 0
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.wd)
    print("learning_rate:{}".format(args.lr))
    for e in range(args.epochs):
        train_epoch(model, e, train_loader, optimizer)
        # test_epoch(model, train_loader, e)
        acc, cluster_score = test_epoch(model, train_loader, e)
        if best_acc < acc:
            best_acc = acc
            best_epoch = e
            best_cluster_score = cluster_score
            torch.save(model.state_dict(), './Result/'+args.dataset)


    print("best epoch:{}, ACC: {}, Cluster Score: {}".format(best_epoch, best_acc, best_cluster_score))
    return best_acc



if __name__ == '__main__':
    starttime = datetime.now()
    parser = argparse.ArgumentParser(description='Adaptive clustering algorithm based on symbolization and large language model')

    # Dataset parameters
    parser.add_argument('--train_path', default='./datas/mushroom/test.csv', help='dataset path')
    parser.add_argument('--dataset', type=str, default='mushroom.pth', help='dataset name')
    parser.add_argument('--feature_list', default='./datas/mushroom/list.txt', help='feature list.txt path')
    parser.add_argument('--feature_dim', default=117, type=int, help='the feature num of data point')
    parser.add_argument('--batch_size', type=int, default=256, help='the number for a batch')

    # Large Language model parameters
    parser.add_argument('--lm_path', default='./bert_models/BertForMaskedLM', help='Large Language Model Path')
    parser.add_argument('--token_flag', default=True, type=bool, help='whether to add additional tokens')
    parser.add_argument('--pretrain', type=bool, default=False, help='whether use pretrained model')

    # model parameters
    parser.add_argument('--epochs', default=50, type=int, help='the number of train epochs')
    parser.add_argument('--cluster_space', default=32, type=int, help='the space for clustering')
    parser.add_argument('--word_emb_dim', default=32, type=int, help='the dimension of word embedding')
    parser.add_argument('--n_res_layers', default=1, type=int, help='the time of residual layers')
    parser.add_argument('--h_dim', default=32, type=int, help='the dimension of hidden layer')

    parser.add_argument('--symbol_space', default=16, type=int, help='the symbol space for a feature')
    parser.add_argument('--lr', default=6e-5, type=float, help='the learning rate of the model')
    parser.add_argument('--wd', default=0.01, type=float, help='the decay of weight')
    parser.add_argument('--alpha', default=1.0, type=float, help='the weight of clustering')


    # device parameters
    parser.add_argument('--gpu_id', default=0, type=int, help='which gpu of the device')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging the train status')

    args = parser.parse_args()
    # add tokens
    if args.token_flag:
        add_additional_tokens(args.lm_path, args.feature_dim)

    # Load data
    feature_map, train_dataset, Fea_num = get_dataset(args.train_path, args.feature_list)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)

    # Main body
    model = UPCD(args)
    model.to(model.device)
    print("model:", model)
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of trainable parameters: {trainable_params}")
    best_acc = solver(args, model, train_loader)
    endtime = datetime.now()
    t = (endtime - starttime).seconds
    print('*************************The total time is ', t)




