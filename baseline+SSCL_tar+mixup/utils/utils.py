import numpy as np
import scipy as sp
# import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import logging
import shutil
import math
import datetime
from OT_torch_ import  cost_matrix_batch_torch, GW_distance_uniform, IPOT_distance_torch_batch_uniform
from sklearn.cluster import SpectralClustering

import torch
import torch.nn as nn
from torchvision import transforms

def same_seeds(seed):
    # 为CPU中设置种子，生成随机数
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 为特定GPU设置种子，生成随机数
        torch.cuda.manual_seed(seed)
        # 为所有GPU设置种子，生成随机数
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    torch.backends.cudnn.benchmark = False 
    # deterministic 置为 True 的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。
    torch.backends.cudnn.deterministic = True

    # 比金恩少一个固定哈希的操作
    # os.environ['PYTHONHASHSEED'] = str(seed)

# 这种初始化方式不好，原因见https://blog.csdn.net/qq_37297763/article/details/116430049
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.xavier_uniform_(m.weight, gain=1)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif classname.find('BatchNorm') != -1:
#         if m.weight is not None: # 防止 affine=False 导致 weight == none, bias == none
#             nn.init.normal_(m.weight, 1.0, 0.02)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif classname.find('Linear') != -1:
#         nn.init.xavier_normal_(m.weight)
#         if m.bias is not None:
#             m.bias.data = torch.ones(m.bias.data.size())

# 推荐的初始化方式
def weights_init(m):
    if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None: # 防止 affine=False 导致 weight == none, bias == none
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

# new init
# def weights_init(m):
#     if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
#         nn.init.xavier_uniform_(m.weight, gain=1)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
#         if m.weight is not None: # 防止 affine=False 导致 weight == none, bias == none
#             nn.init.normal_(m.weight, 1.0, 0.02)
#         if m.bias is not None:
#             m.bias.data.zero_()
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0.0, std=0.01)
#         if m.bias is not None:
#             nn.init.normal_(m.bias, mean=0.0, std=0.01)

# hwq
# def weights_init(m):
#     if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
#         nn.init.normal_(m.weight, mean=0.0, std=0.01)
#         if m.bias is not None:
#             nn.init.normal_(m.bias, mean=0.0, std=0.01)
#     elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
#         nn.init.normal_(m.weight, mean=1.0, std=0.01)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0, std=0.01)
#         if m.bias is not None:
#             nn.init.normal_(m.bias, mean=0, std=0.01)

# def mean_confidence_interval(data, confidence=0.95):
#     a = 1.0*np.array(data)
#     n = len(a)
#     m, se = np.mean(a), scipy.stats.sem(a)
#     h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
#     return m,h

from operator import truediv

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

import torch.utils.data as data

class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()
        #        if medicinal==4 and d==2:
        #            self.train_data=self.imdb['data'][self.x1,:]
        #            self.train_labels=self.imdb['Labels'][self.x1]
        #            self.test_data=self.imdb['data'][self.x2,:]
        #            self.test_labels=self.imdb['Labels'][self.x2]

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            # all_good[class_] = all_set[class_][:200]
            all_good[class_] = all_set[class_][len(all_set[class_])-200:] # 比如有300个，直接截断后200个。
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def sanity_check_unlabel(all_set, num_unlabel):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        # all_good[class_] = all_set[class_][:200]
        all_good[class_] = all_set[class_][len(all_set[class_])-num_unlabel:]
        nclass += 1
        nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0] # paviaU
    label_key = label_file.split('/')[-1].split('.')[0] # paviaU_gt
    data_all = image_data[data_key] # dic-> narray , UP:ndarray(610,340,103)
    GroundTruth = label_data[label_key] # (610, 340)

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:])) # (207400,103)
    data_scaler = preprocessing.scale(data.astype(float))  # (X-X_mean)/X_std,对列进行标准化，即对所有样本光谱的每一个波段标准化
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2]) # 标准化之后再变回去

    return Data_Band_Scaler, GroundTruth

def load_data_houston(image_file, label_file,label_file1):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    label_data1 = sio.loadmat(label_file1)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    label_key1 = label_file1.split('/')[-1].split('.')[0]

    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth_train = label_data[label_key]
    GroundTruth_test = label_data1[label_key1]


    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth_train, GroundTruth_test  # image:(512,217,3),label:(512,217)

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

def plot_embedding_2D(data, label, title, color_map):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min) # 去掉就跟彭老师那个比较像了
    fig = plt.figure()
    for i in range(data.shape[0]): # 彭老师是这种按照测试次序画的每一个点，构成覆盖关系；如果按照类别画总感觉有些奇怪
        # plt.scatter(data[i, 0], data[i, 1], marker='o', color=color_map[label[i]]) # 不放label属性，要不图例会渲染所有的点，而且没法控制次序。markersize控制点的大小
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=4, color=color_map[label[i]]) # 不放label属性，要不图例会渲染所有的点，而且没法控制次序

    # 为了展示图例
    for i in range(len(color_map)):
        plt.scatter([], [], marker='o', color=color_map[i], label=i+1) # 表示散点
        # plt.plot([], [], marker='o', color=color_map[i], label=i+1) # 表示线条

    # plot表示线条画图，所以图例是贯穿圆点的线； scatter表示点画图，所以图例是点。 scatter当点多的时候，效率远不如plot
    # 用plt.plot的线条画散点图，速度快，速度差5倍；用plt.scatter的散点作为依托，加label，用于展示图例，如果是用plot则展示的是贯穿圆点的线。
    # 这样既保证了时间又保证了图例显示正常。

    # plt.xticks([]) # 不显示横轴坐标值
    # plt.yticks([]) # 不显示纵轴坐标值
    plt.title(title)
    # 由于legend是一个方框，bbox_to_anchor=(num1, num2)相当于表示一个点，那么legend的哪个位置位于这个点上呢。参数num3就用以表示哪个位置位于该点。
    # loc 默认值是best，当点多的时候很慢；loc表示图例的哪个点作为anchor点，这里选择左下角的点。bbox_to_anchor，表示相对于anchor的位置。borderaxespad表示填充。prop = {'size':4}表示图例大小
    # plt.legend(bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0, prop = {'size':8})
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0, prop = {'size':8})

    return fig

def allocate_tensors():
    """
    init data tensors
    :return: data tensors
    """
    tensors = dict()
    tensors['support_data'] = torch.FloatTensor()
    tensors['support_label'] = torch.LongTensor()
    tensors['query_data'] = torch.FloatTensor()
    tensors['query_label'] = torch.LongTensor()
    return tensors

def allocate_tensors_unlabel():
    """
    init data tensors
    :return: data tensors
    """
    tensors = dict()
    tensors['support_data'] = torch.FloatTensor()
    tensors['support_label'] = torch.LongTensor()
    tensors['query_data'] = torch.FloatTensor()
    # tensors['query_label'] = torch.LongTensor()
    return tensors

def set_tensors(tensors, batch):
    """
    set data to initialized tensors
    :param tensors: initialized data tensors
    :param batch: current batch of data
    :return: None
    """
    support_data, support_label, query_data, query_label = batch
    tensors['support_data'].resize_(support_data.size()).copy_(support_data)
    tensors['support_label'].resize_(support_label.size()).copy_(support_label)
    tensors['query_data'].resize_(query_data.size()).copy_(query_data)
    tensors['query_label'].resize_(query_label.size()).copy_(query_label)

def set_tensors_unlabel(tensors, batch):

    support_data, support_label, query_data = batch
    tensors['support_data'].resize_(support_data.size()).copy_(support_data)
    tensors['support_label'].resize_(support_label.size()).copy_(support_label)
    tensors['query_data'].resize_(query_data.size()).copy_(query_data)
    # tensors['query_label'].resize_(query_label.size()).copy_(query_label)

def save_checkpoint(state, is_best, exp_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if DPGN model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'),
                        os.path.join('{}'.format(exp_name), 'model_best.pth.tar'))


def adjust_learning_rate(optimizers, lr, iteration, dec_lr_step, lr_adj_base):
    """
    adjust learning rate after some iterations
    :param optimizers: the optimizers
    :param lr: learning rate
    :param iteration: current iteration
    :param dec_lr_step: decrease learning rate in how many step
    :return: None
    """
    # new_lr = lr * (lr_adj_base ** (int(iteration / dec_lr_step)))
    new_lr = lr / math.pow((1 + 10 * (iteration - 1) / dec_lr_step), 0.75)
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def label2edge(label, device):
    """
    convert ground truth labels into ground truth edges
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth edges
    :return: ground truth edges
    """
    # get size
    num_samples = label.size(1)
    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)
    # compute edge
    edge = torch.eq(label_i, label_j).float().to(device)
    return edge

def one_hot_encode(num_classes, class_idx, device):
    """
    one-hot encode the ground truth
    :param num_classes: number of total class
    :param class_idx: belonging class's index
    :param device: the gpu device that holds the one-hot encoded ground truth label
    :return: one-hot encoded ground truth label
    """
    return torch.eye(num_classes)[class_idx].to(device)


def preprocess(num_ways, num_shots, num_queries, batch_size, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots # 9 * 1 = 9
    num_samples = num_supports + num_queries * num_ways #  9 * 1 + 19 * 9 = 180

    # set edge mask (to distinguish support and query edges) 设置边掩码（用于区分支持和查询边）
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device) # 作用？mask for unlabeled data (for semi-supervised setting)
    return num_supports, num_samples, query_edge_mask, evaluation_mask

def preprocess_one(num_supports, num_samples, batch_size, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set size of support set, query set and total number of data in single task

    # set edge mask (to distinguish support and query edges)
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device)
    return num_supports, query_edge_mask, evaluation_mask

def initialize_nodes_edges(batch, num_supports, tensors, batch_size, num_queries, num_ways, device):
    """
    :param batch: data batch
    :param num_supports: number of samples in support set
    :param tensors: initialized tensors for holding data
    :param batch_size: how many tasks per batch
    :param num_queries: number of samples in query set
    :param num_ways: number of classes for each few-shot task
    :param device: the gpu device that holds all data

    :return: data of support set,
             label of support set,
             data of query set,
             label of query set,
             data of support and query set,
             label of support and query set,
             initialized node features of distribution graph (Vd_(0)),
             initialized edge features of point graph (Ep_(0)),
             initialized edge_features_of distribution graph (Ed_(0))
    """
    # allocate data in this batch to specific variables
    set_tensors(tensors, batch)
    support_data = tensors['support_data'].squeeze(0)
    support_label = tensors['support_label'].squeeze(0)
    query_data = tensors['query_data'].squeeze(0)
    query_label = tensors['query_label'].squeeze(0)

    # initialize nodes of distribution graph
    node_gd_init_support = label2edge(support_label, device)
    node_gd_init_query = (torch.ones([batch_size, num_queries, num_supports])
                          * torch.tensor(1. / num_supports)).to(device)
    node_feature_gd = torch.cat([node_gd_init_support, node_gd_init_query], dim=1)

    # initialize edges of point graph
    all_data = torch.cat([support_data, query_data], 1)
    all_label = torch.cat([support_label, query_label], 1)
    all_label_in_edge = label2edge(all_label, device)
    edge_feature_gp = all_label_in_edge.clone()

    # uniform initialization for point graph's edges
    edge_feature_gp[:, num_supports:, :num_supports] = 1. / num_supports
    edge_feature_gp[:, :num_supports, num_supports:] = 1. / num_supports
    edge_feature_gp[:, num_supports:, num_supports:] = 0
    for i in range(num_queries):
        edge_feature_gp[:, num_supports + i, num_supports + i] = 1

    # initialize edges of distribution graph (same as point graph)
    edge_feature_gd = edge_feature_gp.clone()

    return support_data, support_label, query_data, query_label, all_data, all_label_in_edge, \
           node_feature_gd, edge_feature_gp, edge_feature_gd

def unlabel2edge(data, device):
    """
    convert ground truth labels into ground truth edges
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth edges
    :return: ground truth edges
    """
    # get size
    num_samples = data.size(1)
    # reshape
    scores = torch.einsum('bhm,bmn->bhn', data, data.transpose(2,1))
    edge = torch.nn.functional.softmax(scores, dim=-1)
    return edge

def initialize_nodes_edges_unlabel(batch, num_supports, tensors, batch_size, num_queries, num_ways, device):

    # allocate data in this batch to specific variables
    set_tensors_unlabel(tensors, batch)
    support_data = tensors['support_data'].squeeze(0)
    support_label = tensors['support_label'].squeeze(0)
    query_data = tensors['query_data'].squeeze(0)
    # query_label = tensors['query_label'].squeeze(0)

    # initialize nodes of distribution graph

    node_gd_init_support = label2edge(support_label, device)
    node_gd_init_query = (torch.ones([batch_size, num_queries, num_supports])
                          * torch.tensor(1. / num_supports)).to(device)
    node_feature_gd = torch.cat([node_gd_init_support, node_gd_init_query], dim=1)

    # initialize edges of point graph
    all_data = torch.cat([support_data, query_data], 1)
    all_label_in_edge = unlabel2edge(all_data, device)
    edge_feature_gp = all_label_in_edge.clone()

    # uniform initialization for point graph's edges
    edge_feature_gp[:, num_supports:, :num_supports] = 1. / num_supports
    edge_feature_gp[:, :num_supports, num_supports:] = 1. / num_supports
    edge_feature_gp[:, num_supports:, num_supports:] = 0
    for i in range(num_queries):
        edge_feature_gp[:, num_supports + i, num_supports + i] = 1

    # initialize edges of distribution graph (same as point graph)
    edge_feature_gd = edge_feature_gp.clone()

    return support_data, query_data, all_data, all_label_in_edge, \
           node_feature_gd, edge_feature_gp, edge_feature_gd

def OT(src, tar, ori=False, sub=False, **kwargs):
    wd, gwd = [], []
    for i in range(len(src)): # (1, 128, 180)
        source_share, target_share = src[i], tar[i]
        cos_distance = cost_matrix_batch_torch(source_share, target_share)
        cos_distance = cos_distance.transpose(1,2)
        # TODO: GW as graph matching loss
        beta = 0.1
        if sub:
            cos_distance = kwargs['w_st']*cos_distance
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = torch.nn.functional.relu(cos_distance - threshold)
        
        wd_val = - IPOT_distance_torch_batch_uniform(cos_dist, source_share.size(0), source_share.size(2), target_share.size(2), iteration=30)
        gwd_val = GW_distance_uniform(source_share, target_share, sub,**kwargs)
        wd.append(abs(wd_val))
        gwd.append(abs(gwd_val))

    ot = sum(wd)/len(wd) + sum(gwd)/len(gwd)
    return ot, sum(wd)/len(wd), sum(gwd)/len(gwd)

# old log Ubuntu下可用，但是win10下不能写入到文件，因为myTimeFormat里的那个冒号，和后边的.txt没有关系。
# def set_logging_config(logdir, num_seeds):
#     myTimeFormat = '%Y-%m-%d_%H:%M'
#     nowTime = datetime.datetime.now().strftime(myTimeFormat)  # 有strftime约束之后是字符串格式，否则是datetime格式
#
#     if not os.path.exists(logdir):
#         os.makedirs(logdir)
#     logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", # 加了个levelname
#                         level=logging.INFO,
#                         handlers=[logging.FileHandler(os.path.join(logdir, str(num_seeds) +'seeds_'+nowTime+'_log.txt')),# 加了个时间
#                                   logging.StreamHandler(os.sys.stdout)])

# new log %S代表秒，%s代表字符串
def set_logging_config(logdir, num_seeds):
    myTimeFormat = '%Y-%m-%d_%H-%M-%S'
    nowTime = datetime.datetime.now().strftime(myTimeFormat)  # 有strftime约束之后是字符串格式，否则是datetime格式

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s", # 加了个levelname
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, str(num_seeds) +'seeds_'+nowTime+'.log')),# 加了个时间
                                  logging.StreamHandler(os.sys.stdout)])

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def cal_clsvec_init(data, fine_labels, num_class):
    class_vec = np.zeros([num_class, data.shape[1]])
    for i in range(num_class):
        idx = [j for j, x in enumerate(fine_labels) if x == i]
        sigma_cls = np.zeros([data.shape[0], data.shape[1]])
        for m in range(len(idx)):
            s = data[:, :, idx[m]]
            avg_s = sum(s) / len(s)
            sigma_cls += avg_s # 广播
        vec = sum(sigma_cls) / len(idx)
        # 输出每类的向量表示
        class_vec[i] = vec

    return class_vec

# 从原始数据获取精细类到粗糙类的知识结构
def gen_superclass(raw_data, fine_labels, num_class, num_clusters):
    '''
        data.shape = (采样时间点， 信号, batch_size)
        fine_labels = (batch_size, )
    '''
    # HSI数据和论文的故障数据类型不一致，需要转化：
    # raw_data = (3600,128,9,9) -> (3600,128,9*9) -> (9*9,128,3600)
    n, c, _, _ = raw_data.shape
    data = raw_data.reshape(n, c, -1)
    data = np.transpose(data, (2, 1, 0))
    class_vec = cal_clsvec_init(data, fine_labels, num_class)
    aff_mat = np.zeros([num_class, num_class])
    for a in range(0, num_class - 1):
        for b in range(a + 1, num_class):
            # l2-范数
            distance = np.linalg.norm(class_vec[a] - class_vec[b])
            aff_mat[a, b] = distance
            aff_mat[b, a] = aff_mat[a, b]
    beta = 0.1
    aff_mat = np.exp(-beta * aff_mat / aff_mat.std())
    for i in range(num_class):
        aff_mat[i, i] = 0.0001
    # ‘precomputed’:将X解释为预先计算的亲和矩阵。如果affinity='precomputed'，那这步输入的就是表征实例间相似性的亲和矩阵。
    sc = SpectralClustering(num_clusters, affinity='precomputed', assign_labels='discretize')  # discretize离散化，受初始化影响小
    groups = sc.fit_predict(aff_mat) # # 返回的是聚类标签ndarray, shape (n_samples,)

    return groups


def get_lambda(num_ep, cur_ep, coarse_train_ep, fine_train_ep):
    if cur_ep > num_ep - fine_train_ep - 1:
        my_lambda = 0
    elif cur_ep < coarse_train_ep:
        my_lambda = 1
    else:
        my_lambda = 1 - ((cur_ep + 1 - coarse_train_ep) / (num_ep - fine_train_ep)) ** 2

    return my_lambda

def gen_coarselabel(fine_labels, relation):
    c_labels = []
    for i in range(len(fine_labels)):
        fl = int(fine_labels[i])
        cl = relation[fl]
        c_labels.append(cl)

    return np.array(c_labels)


def get_scalingfac(num1, num2):
    s1 = int(math.floor(math.log10(num1)))
    s2 = int(math.floor(math.log10(num2)))
    scale = 10 ** (s1 - s2)
    return scale

def mixup_data(x_1, x_2, alpha = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mixed_x = lam * x_1 + (1 - lam) * x_2

    return mixed_x, lam



