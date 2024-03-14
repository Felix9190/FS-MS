from collections import OrderedDict

config = OrderedDict()
config['data_path'] = '../../datasets'
config['save_path'] = '/results'
config['source_data'] = 'Chikusei_imdb_128_7_7.pickle'
config['target_data'] = 'Houston/data.mat'
config['target_data_gt_train'] = 'Houston/mask_train.mat'
config['target_data_gt_test'] = 'Houston/mask_test.mat'
config['num_generation'] = 1 # 对偶图交互次数，默认是1，2效果不好而且时间代价更大
config['gpu'] = 1
config['point_distance_metric'] = 'l2'
config['distribution_distance_metric'] = 'l2'

config['log_dir'] = './logs'

train_opt = OrderedDict()
train_opt['patch_size'] = 7
train_opt['batch_task'] = 1 # ？？
train_opt['num_ways'] = 5 # 没用到都
train_opt['num_shots'] = 1 # 没用到都
train_opt['episode'] = 5000 # 3000 + 1500 + 4500
train_opt['coarse_ep'] = 3000
train_opt['fine_ep'] = 4500
train_opt['lr'] = 1e-3 # 1e-3 = 0.001  # init_lr = args.lr * args.batch_size / 256  args.lr = 0.05 先试下0.05 和 0.05*0.7 = 0.035，不行再余弦退火
train_opt['weight_decay'] = 1e-4 # 0.001
train_opt['dropout'] = 0.1

train_opt['lambda_1'] = 0.1
train_opt['d_emb'] = 128 # 嵌入向量维度
train_opt['src_input_dim'] = 128
train_opt['tar_input_dim'] = 144 # Houston2013
train_opt['n_dim'] = 100
train_opt['src_class_num'] = 18 # 源域中大于200个的类别数

train_opt['shot_num_per_class'] = 1
train_opt['query_num_per_class'] = 19

train_opt['tar_class_num'] = 15 # 以目标域类别作为元任务类别数。
train_opt['tar_lsample_num_per_class'] = 5

# CL-block
# train_opt['hid_units'] = 32 # 也可以64/512尝试一下

# C_F_Classifier
train_opt['hidden_channels'] = 64 # 也可以32

config['train_config'] = train_opt

