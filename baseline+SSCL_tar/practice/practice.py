import numpy as np
import torch
import random

# data = {13:2, 2:4, 4:8}
# data = sorted(list(data)) # 输出的是key值
# print(data)

# list1 = [1,2,3,4,5,6,7,8,9]
# samples = random.sample(list1, len(list1))
# print(samples)
# random.shuffle(samples)
# print(samples)

# data = np.arange(12)
# data = data.reshape(-1, 2, 3)
# print(data.shape)
# print(type(data.shape))
# print(data.shape[0])
# print(data.shape[2])
# print(data.shape[1])
# print(data)

import numpy as np
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler


# acc = np.array([82.64725843, 84.26435141,76.37780534,86.25587981,82.02241932]) # UP
# acc = np.array([69.9282132, 69.75120464, 65.47349789, 73.94040712, 76.16284787]) # IP
# acc = np.array([91.99245129,90.76023608,92.36803641,90.25328868,86.66765343]) # Salinas

# acc = [82.32898832,81.89604737,80.27661417,81.58479792,84.60602373] # Gia-CFSL UP
# acc = [64.95230603,67.04690727,56.70174058,64.58845511,65.17848363] # Gia-CFSL IP
# acc = [81.91944958,81.15185697,81.29929091,83.87821488,83.50611968]
# acc = [85.09747022, 83.98118462, 88.46270857, 87.65767242,87.48449603]
# acc = [88.31761485,82.64257799,87.58278533,86.59521191,86.13886874,89.63749971,82.22133814,87.25281412]

# acc = [88.31761485,82.64257799,87.58278533,86.59521191,86.13886874,89.63749971,82.22133814,87.25281412]
# # acc = [91.69272327,92.19226998,91.80188348,89.33190253,89.43921257,90.9508039 ] # average OA: 90.90 +- 1.13
# acc = [75.25813748,76.35952404,76.80204543,78.09027436,72.36699774,75.53348412]
# acc = [77.73625725, 82.56465729, 80.24387845,82.66299538,74.70744419,82.11230209, 79.51617662, 81.54194119,82.82033632]
# acc = [76.73320877,80.83390697,80.21437703, 83.60704101, 75.25813748]
acc = [75.60055751, 71.79634336, 75.59235878]
OAMean = np.mean(acc)
OAStd = np.std(acc)
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))

print(np.arange(1211, 1241))

# import torch
# from transformers import BertTokenizer, BertModel
#
# # 加载预训练的BERT模型和分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # 将标签列表转换为Tensor
# labels_tensor = torch.tensor(labels_tar)
#
# # 编码单个单词的特征向量
# def encode_word(word):
#     # 使用BERT的分词器对单词进行编码
#     input_ids = tokenizer.encode(word, add_special_tokens=False)
#     input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)  # 添加批次维度
#
#     # 使用BERT模型获取特征向量
#     with torch.no_grad():
#         outputs = model(input_ids_tensor)
#         encoded_word = torch.mean(outputs.last_hidden_state, dim=1)  # 取平均作为特征向量
#
#     return encoded_word
#
# # 编码整个标签的特征向量并计算平均值
# def encode_label(label):
#     words = label.split("-")  # 使用"-"分割多个单词
#     encoded_words = [encode_word(word) for word in words]  # 对每个单词进行编码
#     encoded_label = torch.mean(torch.stack(encoded_words), dim=0)  # 计算编码后的单词向量的平均值
#     return encoded_label
#
# # 对标签进行编码并计算平均值
# encoded_labels = [encode_label(label) for label in labels_tar]
# semantic_vectors = torch.stack(encoded_labels)
#
# print(semantic_vectors)


# 加载目标域数据
# import os
# from utils import utils

# test_data = os.path.join('../../../datasets','houston2013/Houston.mat')  # (349, 1905, 144)
# test_label = os.path.join('../../../datasets','houston2013/Houston_gt.mat')  # (349, 1905)

# test_data = os.path.join('../../../datasets','Houston/data.mat')  # (349, 1905, 144)
# test_label = os.path.join('../../../datasets','Houston/mask_test.mat')  # (349, 1905)
#
# Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)
#
# data = GroundTruth
# mask = np.unique(data)
# labeled_pixels = 0
# sum_pixels = 0
# tmp = {}
# for v in mask:
#     tmp[v] = np.sum(data == v)
#     if v > 0:
#         labeled_pixels += tmp[v]
#     sum_pixels += tmp[v]
# print(mask) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
# print(tmp) # {0: 649816, 1: 1251, 2: 1254, 3: 697, 4: 1244, 5: 1242, 6: 325, 7: 1268, 8: 1244, 9: 1252, 10: 1227, 11: 1235, 12: 1233, 13: 469, 14: 428, 15: 660}
# print(labeled_pixels) # 15029
# print(sum_pixels)
#
# print("OK")

# mask_train
# {0: 662013, 1: 198, 2: 190, 3: 192, 4: 188, 5: 186, 6: 182, 7: 196, 8: 191, 9: 193, 10: 191, 11: 181, 12: 192, 13: 184, 14: 181, 15: 187}
# 2832
# 664845

# mask_test
# {0: 652648, 1: 1053, 2: 1064, 3: 505, 4: 1056, 5: 1056, 6: 143, 7: 1072, 8: 1053, 9: 1059, 10: 1036, 11: 1054, 12: 1041, 13: 285, 14: 247, 15: 473}
# 12197  81.15643%
# 664845









class MetaTrainDataset(Dataset):
    def __init__(self, image_datas):
        self.image_datas = image_datas

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        return image

def get_metatrain_data_loader():
    domain_specific_metatrain_data = [i for i in range(10)]
    dataset = MetaTrainDataset(domain_specific_metatrain_data)
    loader = DataLoader(dataset, batch_size = 4)

    return loader

# loader = get_metatrain_data_loader()
# episode = 100
# source_iter = iter(loader)

# 这种写法是正确的
# for i in range(episode):
#     try:
#         source_data = source_iter.next()
#         print(source_data)
#     except Exception as err:
#         source_iter = iter(loader)
#         source_data = source_iter.next()
#         print(source_data)

# 错误写法
# train_cl = loader.__iter__().next() # 之前程序写错了！！！！！！！！！！！而且shuffle应该设为true
# print(train_cl) # 当loader的shuffle=false时，全是tensor([0, 1])；=true时候，会输出不一样的



# for data in loader:
#     print(data)
# tensor([0, 1])
# tensor([2, 3])
# tensor([4, 5])
# tensor([6, 7])
# tensor([8, 9])

# np1 = np.random.uniform(1,10,(3,2))
# np2 = np.random.uniform(1,10,(3,2))
# list1 = []
# list1.append(np1)
# list1.append(np2)
# np3 = np.array(list1)
# print(np3)
# print(np3.shape)

# # Import required package
# import numpy as np
#
# # Creating a Dictionary
# # with Integer Keys
# dict = {1: 'Geeks',
#         2: 'For',
#         3: 'Geeks'}
#
# # to return a group of the key-value
# # pairs in the dictionary
# result = dict.items()
# for i in result:
#     print(i)
#
# # Convert object to a list
# data = list(result)
#
# # Convert list to an array
# numpyArray = np.array(data)
#
# # print the numpy array
# print(numpyArray)

# list1 = [1,2,3]
# list2 = [4,5]
# list1 += list2
# print(list1)

# list = []
# a = np.array([6,7,8,9])
# b = np.array([8,9,19,19])
# list.append(a)
# list.append(b)
# print(list)
# list = np.array(list)
# print(list)
# print(list.shape)

def gaussian_noise(image, mean=0, sigma=0.15):
    # mean=0, sigma=0.15
    # mean=0, sigma=0.1
    # mean=0, sigma=0.05
    # 1.可以采用同样的mean和sigma增强，因为即使相同的种子，在一次迭代中顺序随机出的数字也不同
    # 2.也可以采用不同的mean和sigma增强

    """
    添加高斯噪声
    :param image:原图
    param mean:均值
    :param sigma:标准差 值越大，噪声越多
    :return:噪声处理后的图片
    """

    image = np.asarray(image, dtype=np.float32)
    max = np.max(image)
    min = np.min(image)
    length = max - min
    image = (image - min) / length
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = output * length + min
    print("output.max = ", np.max(output))
    print("output.min = ", np.min(output))
    print("output.mean = ", np.mean(output))
    # output = np.clip(output, 0, 1)
    # output = np.uint8(output * length)
    return output




# def gaussian_noise(image, mean=0, sigma=0.15):
#     # mean=0, sigma=0.15
#     # mean=0, sigma=0.1
#     # mean=0, sigma=0.05
#
#     """
#     添加高斯噪声
#     :param image:原图
#     param mean:均值
#     :param sigma:标准差 值越大，噪声越多
#     :return:噪声处理后的图片
#     """
#     # in code ,np.random is up to seed, so need different sigma value , example sigma = 0.1 0.05 0.15
#
#     image = np.asarray(image, dtype=np.float32)
#     max = np.max(image)
#     min = np.min(image)
#     length = max - min
#     print("max = ", max)
#     print("min = ", min)
#     print("length = ", length)
#     print("image.mean = ", np.mean(image))
#     image = (image - min) / length
#     # image = np.asarray((image - min) / length, dtype=np.float32)  # 图片灰度标准化 0-1
#
#     noise1 = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
#     noise2 = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
#     output1 = image + noise1  # 将噪声和图片叠加
#     output2 = image + noise2
#     output1 = output1 * length + min
#     output2 = output2 * length + min
#     # output = np.clip(output, 0, 1)
#     # output = np.uint8(output * length)
#     print("output1.max = ", np.max(output1))
#     print("output1.min = ", np.min(output1))
#     print("output1.mean = ", np.mean(output1))
#     print("output2.max = ", np.max(output2))
#     print("output2.min = ", np.min(output2))
#     print("output2.mean = ", np.mean(output2))
#     return output1, output2

# torch.manual_seed(0)
# np.random.seed(0)
# supports_src = torch.randn([3,6,3,3]).to(0)
# supports_src = supports_src * 30
# # print(supports_src.device)
# # print(type(supports_src))
# # print(supports_src)
# supports_src1 = torch.FloatTensor(gaussian_noise(supports_src.data.cpu())).to(0)
# supports_src2 = torch.FloatTensor(gaussian_noise(supports_src.data.cpu())).to(0)




# print("-------------------------------------------------------------")
# print(supports_src.shape)
# print(supports_src.device)
# print(type(supports_src))
# print(supports_src1)
# print(supports_src2)

# np.random.seed(0)
# a = np.random.randn(10)
# b = np.random.randn(10)
# c = np.random.randn(10)
# print("a = ", a)
# print("b = ", b)
# print("c = ", c)
# 可以看到三次的结果是不一样的，np.random.seed(0)是保证每次运行程序时和上次的结果一样，第一个np.random.randn(10)会随机出的数字是确定等于上次的，第二次会随机出的数字是确定等于上次的，但不等于第一个
# output
# a =  [ 1.76405235  0.40015721  0.97873798  2.2408932   1.86755799 -0.97727788
#   0.95008842 -0.15135721 -0.10321885  0.4105985 ]
# b =  [ 0.14404357  1.45427351  0.76103773  0.12167502  0.44386323  0.33367433
#   1.49407907 -0.20515826  0.3130677  -0.85409574]
# c =  [-2.55298982  0.6536186   0.8644362  -0.74216502  2.26975462 -1.45436567
#   0.04575852 -0.18718385  1.53277921  1.46935877]



# source
# CH  include background. data.max =  15133.0  data.min =  0.0 data_scaler.max =  88.62107407294518 data_scaler.min =  -2.511908360528769
# CH  data_train.max =  32.416466  data_train.min =  -2.352965
# UP  data.max =  8000  data.min =  0         Data_Band_Scaler.max =  15.918664053448387   Data_Band_Scaler.min =  -2.8023924636727973
# Salinas  data.max =  9207  data.min =  -11  Data_Band_Scaler.max =  222.7637925583525    Data_Band_Scaler.min =  -6.272498288061055
# IP  data.max =  9604  data.min =  955        Data_Band_Scaler.max =  8.992655544253978   Data_Band_Scaler.min =  -7.642090293989591




