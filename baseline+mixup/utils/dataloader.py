import numpy as np
import random

import torch
from torch.utils.data import DataLoader, Dataset


from torch.utils.data.sampler import Sampler

class Task(object):
    # 把源域里得到的18个类的metatrain_data，随机选取9个类，再分支持集和查询集。
    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data)) # 这个排序也没有意义，反正都要随机采样，没必要排序
        class_list = random.sample(class_folders, self.num_classes) # 从18个类别随机采样9个类别
        labels = np.array(range(len(class_list)))
        labels = dict(zip(class_list, labels))
        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []

        self.support_real_labels = []
        self.query_real_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c]) # 为什么要打乱两次, 感觉毫无意义

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)] # 赋予一个新的类别0-8
            self.query_labels += [labels[c] for i in range(query_num)]

            self.support_real_labels += [c for i in range(shot_num)] # origin classes
            self.query_real_labels += [c for i in range(query_num)] # origin classes

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和query
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader


'''
    用于自监督对比学习的不带标签的训练数据集（元任务就从这里边抽取）
    CH源域：每类200个，共18类，3600个样本，batchsize = 256, 15次迭代一个epoch，1000个episode大概66.67个epoch
    UP目标域：每类200个，共9类，1800个样本，batchsize = 128, 15次迭代一个epoch，2000个episode大概133，33个epoch
'''
# def getMetaTrainDataset(metatrain_data, nums_class) :
#     domain_specific_metatrain_data = []
#     for c in range(nums_class):
#         domain_specific_metatrain_data += metatrain_data[c]
#         random.shuffle(domain_specific_metatrain_data)
#     return domain_specific_metatrain_data
#
# class MetaTrainDataset(Dataset):
#     def __init__(self, image_datas):
#         self.image_datas = image_datas
#
#     def __len__(self):
#         return len(self.image_datas)
#
#     def __getitem__(self, idx):
#         image = self.image_datas[idx]
#         return image

# def get_metatrain_data_loader(metatrain_data, nums_class, split='src'):
#     domain_specific_metatrain_data = getMetaTrainDataset(metatrain_data, nums_class)
#     dataset = MetaTrainDataset(domain_specific_metatrain_data)
#     if split == 'src':
#         loader = DataLoader(dataset, batch_size = 256)
#     else:
#         loader = DataLoader(dataset, batch_size = 128)
#
#     return loader


'''
    源域元训练带标签数据集（元任务就从这里边抽取）
    CH源域：每类200个，共18类，3600个样本，batchsize = 256, 15次迭代一个epoch，1000个episode大概66.67个epoch
'''
def getMetaTrainLabeledDataset(metatrain_data) :
    # metatrain_data dict 18 里边key是类别，value是list 200，[9, 9, 128]
    src_metatrain_data = []
    src_metatrain_label = []
    for c in range(len(metatrain_data)):
        src_metatrain_data.append(np.array(metatrain_data[c])) 
        src_metatrain_label.append(np.full(len(metatrain_data[c]), c))
    # (18,200,9,9,128)
    src_metatrain_data = np.array(src_metatrain_data)
    _, _, w, h, c = src_metatrain_data.shape
    # (3600,9,9,128)
    src_metatrain_data = src_metatrain_data.reshape(-1, w, h, c)
    # (3600,128,9,9)
    src_metatrain_data = np.transpose(src_metatrain_data, (0, 3, 1, 2))
    # (18,200)
    src_metatrain_label = np.array(src_metatrain_label)
    # (3600,)
    src_metatrain_label = src_metatrain_label.reshape(-1)
    # 之前那个对比的程序也要删掉这个
    # random.shuffle(domain_specific_metatrain_data)
    return src_metatrain_data, src_metatrain_label

class MetaTrainLabeledDataset(Dataset):
    def __init__(self, image_datas, image_labels):
        self.image_datas = image_datas
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.image_labels[idx]
        return image, label

def get_metatrain_Labeled_data_loader(src_metatrain_data, src_metatrain_label):
    dataset = MetaTrainLabeledDataset(src_metatrain_data, src_metatrain_label)
    loader = DataLoader(dataset, batch_size = 256, shuffle=True)
    return loader


from . import utils, data_augment
import math

# 看看能不能优化一下，这个loader都是干嘛的？
# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, tar_lsample_num_per_class, shot_num_per_class, HalfWidth):

    print(Data_Band_Scaler.shape)  # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    # HalfWidth
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {}  # Data Augmentation
    m = int(np.max(G))
    nlabeled = tar_lsample_num_per_class  # 5   nlabeled和shot_num_per_class都是通过TAR_LSAMPLE_NUM_PER_CLASS赋值的，相等。
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)  # 40
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520 应该不是这个结果，输出检查！
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth : Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth : Column[RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        # imdb_da_train['data'][:, :, :, iSample] = data_augment.Crop_and_resize_single(
        #     data[Row[da_RandPerm[iSample]] - HalfWidth: Row[da_RandPerm[iSample]] + HalfWidth + 1,
        #     Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :], HalfWidth)
        imdb_da_train['data'][:, :, :, iSample] = data_augment.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth: Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])

        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)
    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-10 0-9
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, tar_lsample_num_per_class, shot_num_per_class, patch_size):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=class_num,
        tar_lsample_num_per_class=tar_lsample_num_per_class,
        shot_num_per_class=shot_num_per_class,
        HalfWidth=patch_size // 2)
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # 换坐标轴 (9,9,103, 1800)->(1800, 103, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification 和之前的区别就是，把多维数组按类别划分为字典。
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain


def get_target_dataset_houston(Data_Band_Scaler, GroundTruth_train, GroundTruth_test, class_num, tar_lsample_num_per_class, shot_num_per_class, patch_size):
    train_loader, _, imdb_da_train, _, _, _, _, _ = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth_train,
        class_num=class_num,
        tar_lsample_num_per_class=tar_lsample_num_per_class,
        shot_num_per_class=shot_num_per_class,
        HalfWidth=patch_size // 2)
    test_loader, G, RandPerm, Row, Column, nTrain = get_alltest_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth_test,
        class_num=class_num,
        shot_num_per_class=0,
        HalfWidth=patch_size // 2)


    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth_train, GroundTruth_test

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # 换坐标轴 (9,9,103, 1800)->(1800, 103, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification 和之前的区别就是，把多维数组按类别划分为字典。
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain

def get_alltest_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, HalfWidth):

    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)  # (1830, 1020, 103)
    groundtruth = utils.flip(GroundTruth)  # (1830, 1020)

    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth,
        nColumn - HalfWidth:2 * nColumn + HalfWidth]  # (642, 372)
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,
           :]  # (642, 372, 103)

    [Row, Column] = np.nonzero(G)  # (12197,) (12197,) 根据G确定样本所在的行和列
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    max_Row = np.max(Row)
    print('number of sample', nSample)

    train = {}

    m = int(np.max(G))  #

    # 取样
    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]  # G ndarray Row中的索引
        np.random.shuffle(indices)  #
        nb_val = int(len(indices))
        # nb_val = int(proptionVal)
        train[i] = indices[:nb_val]  #

    train_indices = []

    for i in range(m):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of target:', len(train_indices))  # 12197

    nTrain = len(train_indices)  # 12197

    trainX = np.zeros([nTrain,  2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand], dtype=np.float32)
    trainY = np.zeros([nTrain], dtype=np.int64)

    RandPerm = train_indices
    RandPerm = np.array(RandPerm)
    for i in range(nTrain): # 12197
        trainX[i, :, :, :] = data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :] # 7 7 144
        trainY[i] = G[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    trainX = np.transpose(trainX, (0, 3, 1, 2)) # 12197 7 7 144 -> 12197 144 7 7
    trainY = trainY - 1

    print('all data shape', trainX.shape)
    print('all label shape', trainY.shape)

    test_dataset = MetaTrainLabeledDataset(image_datas=trainX, image_labels=trainY)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    return test_loader, G, RandPerm, Row, Column, nTrain


'''
    先不用了，保持和few-shot的训练样本一致
    
    用于源域全部数据的训练数据集（元任务就从这里边抽取）
    CH源域：77592个样本，batchsize = 256, 77592 / 256 = 304 iter = 1 epoch
    暂定（具体多少看损失曲线）：
    源域训练60个epoch，2:1:3 = 20:10:30，粗粒度20，过渡10，细粒度30。60 * 304 = 18240iter 和few-shot一起训练，18240 episode。
    目标域4760episode，看看损失曲线。
    共18240 + 4760 = 23000,方便测试输出
'''

# class sourceDataset(Dataset):
#     def __init__(self, image_datas, image_labels):
#         self.image_datas = image_datas
#         self.image_labels = image_labels
#
#     def __len__(self):
#         return len(self.image_datas)
#
#     def __getitem__(self, idx):
#         image = self.image_datas[idx]
#         label = self.image_labels[idx]
#         return image, label
#
# def get_source_data_loader(data_train, labels_train):
#     source_Dataset = sourceDataset(data_train, labels_train)
#     loader = DataLoader(source_Dataset, batch_size = 256, shuffle=True)
#
#     return loader






