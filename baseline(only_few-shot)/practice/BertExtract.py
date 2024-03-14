import torch
from transformers import BertModel, BertTokenizer


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

labels_tar = ["Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"]
# labels_tar =
encoded_inputs = tokenizer(labels_tar, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**encoded_inputs)

hidden_states = outputs.last_hidden_state
sentence_embeddings = hidden_states[:, 0, :]
# print("sentence_embeddings = ", sentence_embeddings)
# print(sentence_embeddings.shape)
# print(type(sentence_embeddings)) Tensor

label = torch.range(0, 15)
# print(label)


import t_sne
t_sne.t_SNE(sentence_embeddings, label)



# import numpy as np
# # 两个特征向量数组
# array1 = sentence_embeddings
# array2 = sentence_embeddings
#
# # 计算欧氏距离矩阵
# distances = np.linalg.norm(array1[:, np.newaxis] - array2, axis=2)
#
# print(distances)

# vocab_list = ['apple', 'orange', 'banana', 'grape']
# vocab_dict = {0: "Asphalt", 1: "Meadows", 2: "Gravel", 3: "Trees", 4: "Painted-metal-sheets", 5: "Bare-Soil", 6: "Bitumen", 7: "Self-Blocking-Bricks", 8: "Shadows"}
# # choose_class_list = [8,7,6,5,4,3,2,1,0]
# choose_class_list = [0,1,2,3,4,5,6,7,8]
# semantice_list = []
# for i in range(len(choose_class_list)):
#     semantice_list.append(vocab_dict[choose_class_list[i]])
# print(semantice_list)
#
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# encoded_inputs = tokenizer(semantice_list, padding=True, truncation=True, return_tensors='pt')
#
# with torch.no_grad():
#     outputs = model(**encoded_inputs)
#
#     last_hidden_states = outputs.last_hidden_state
#
#     last_hidden_state = torch.mean(last_hidden_states, dim=1)
#
# print(last_hidden_state.shape)
# # print(outputs.pooler_output.shape)  # (9,768) Tensor
#
# from utils import utils
# print(-1 * utils.euclidean_metric(last_hidden_state, last_hidden_state))
# tensor([[ 0.0000, 47.2638, 24.1703, 34.3213, 62.4200, 52.7310, 47.2364, 75.1452,
#          48.0803],
#         [47.2638,  0.0000, 45.0549, 37.4571, 62.0176, 51.6921, 51.5232, 66.5163,
#          37.0553],
#         [24.1703, 45.0549,  0.0000, 39.7323, 71.4350, 54.1700, 48.7850, 82.9118,
#          51.3106],
#         [34.3213, 37.4571, 39.7323,  0.0000, 53.9948, 50.1130, 50.9199, 67.4307,
#          32.8544],
#         [62.4200, 62.0176, 71.4350, 53.9948,  0.0000, 48.3156, 59.9867, 32.8870,
#          51.7490],
#         [52.7310, 51.6921, 54.1700, 50.1130, 48.3156,  0.0000, 56.2893, 53.2725,
#          62.8595],
#         [47.2364, 51.5232, 48.7850, 50.9199, 59.9867, 56.2893,  0.0000, 69.0343,
#          49.6726],
#         [75.1452, 66.5163, 82.9118, 67.4307, 32.8870, 53.2725, 69.0343,  0.0000,
#          65.3127],
#         [48.0803, 37.0553, 51.3106, 32.8544, 51.7490, 62.8595, 49.6726, 65.3127,
#           0.0000]])

# print(-1 * utils.euclidean_metric(outputs.pooler_output , outputs.pooler_output))
# tensor([[  0.0000,  47.9914,  16.6018,   6.9026,  37.7993,  55.4800,  15.0230,
#           45.7268,  64.0274],
#         [ 47.9914,   0.0000, 109.9489,  77.1458, 119.8877, 168.4123,  47.2631,
#          140.5017,   6.7882],
#         [ 16.6018, 109.9489,   0.0000,   7.3143,  37.1956,  32.4012,  45.0933,
#           32.5510, 128.6005],
#         [  6.9026,  77.1458,   7.3143,   0.0000,  26.8973,  34.8590,  22.6410,
#           28.8988,  94.4170],
#         [ 37.7993, 119.8877,  37.1956,  26.8973,   0.0000,  19.2352,  36.2310,
#            8.5233, 147.2474],
#         [ 55.4800, 168.4123,  32.4012,  34.8590,  19.2352,   0.0000,  62.0320,
#           14.2411, 200.6805],
#         [ 15.0230,  47.2631,  45.0933,  22.6410,  36.2310,  62.0320,   0.0000,
#           51.5062,  65.5761],
#         [ 45.7268, 140.5017,  32.5510,  28.8988,   8.5233,  14.2411,  51.5062,
#            0.0000, 169.5886],
#         [ 64.0274,   6.7882, 128.6005,  94.4170, 147.2474, 200.6805,  65.5761,
#          169.5886,   0.0000]])




# for i in range(len(semantice_list)):
#     # print("vector", i,"=", outputs.last_hidden_state[i])
#     # print(outputs.last_hidden_state[i].shape)
#     print("vector", i, "=", outputs.pooler_output[i])
#     print(outputs.pooler_output[i].shape)


