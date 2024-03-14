import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import process

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))  # 512
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        # seq = (1, 2708, 1433) -> seq_fts = (1, 2708, 512)
        seq_fts = self.fc(seq)
        if sparse:
            # (1, 2708, 512)
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)  # (1, 2708, 512)
        if self.bias is not None:
            out += self.bias  # 512

        return self.act(out)

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1) # 512 512 1
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None): # (1, 512) (1, 2708, 512) (1, 2708, 512)
        c_x = torch.unsqueeze(c, 1) # (1, 1, 512)
        c_x = c_x.expand_as(h_pl) # (1, 2708, 512)  512 cols copy 2708 times
        # (1, 2708)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        # (1, 5416)
        logits = torch.cat((sc_1, sc_2), 1)

        return logits

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation): # 1433 512 prelu
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h) # 512 -> 1

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        # (1, 2708, 512) -- （1, 180, 32）
        h_1 = self.gcn(seq1, adj, sparse) # (1, 2708, 1433) (2708, 2708) true
        # (1, 512) -- (1, 32)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        # (1,2708,512) -- (1, 180, 32)
        h_2 = self.gcn(seq2, adj, sparse)
        # (1,5416) -- (1, 360)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret, h_1, h_2, c

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()



        # model = DGI(ft_size, hid_units, nonlinearity)  # 1433 512 prelu


class CL(nn.Module):
    def __init__(self,emb_size, hid_units, batch_size, activation = 'prelu'):
        super(CL, self).__init__()
        self.dgi = DGI(emb_size, hid_units, activation) # 1433 512。 prelu 需要定义activation 128 ，n_h暂时设置成32，也可以64/512
        self.disc = Discriminator(hid_units)
        self.batch_size = batch_size # 1 train_opt['batch_task'],

    def forward(self, point_nodes, point_similarities, sparse = False):
        # (2708, 1433) --  (1, 180, 128)  --> (180,128) , 后期需要改，万一batch_size不是1，就不能这么用
        # 不要传给下一个变量，传给本身！
        dev = point_nodes[0].device # 试一下同一个dev，梯度还会不会消失？会!
        # tensor.data还是tensor，但不会影响原tensor的梯度传播,不会中断梯度图
        # 因为.cpu().detach().numpy()处理完还要接着转换成tensor类型接着用，如果这样处理的话，被赋值的tensor将没有梯度，当loss回传的时候自然都不会被影响
        # 所以千万千万不要直接对原tensor操作，直接对原tensor的数据域进行操作，自然不会影响到原tensor的梯度传导！！
        point_nodes[0].data = torch.FloatTensor(process.preprocess_features(point_nodes[0].data.squeeze().cpu().detach().numpy())).to(dev).unsqueeze(0)
        # features = process.preprocess_features(point_nodes[0].squeeze().cpu().detach().numpy())
        nb_nodes = point_nodes[0].shape[1]  # 2708 -- 180

        # 没必要加自己对自己的相似度，因为本身就有！！！ 看看loss会不会变化？变化了！！
        # 重写GCN：去掉繁琐的稀疏转换过程
        adj = point_similarities[0].data.squeeze().cpu().detach().numpy() # 同样后期需要改，(1, 180, 180)  --> (180,180)
        adj = process.normalize_adj(adj)
        point_similarities[0].data = torch.FloatTensor(adj).to(dev).unsqueeze(0)

        shuffleIndex = np.random.permutation(nb_nodes) # 2708 disorder indexes -- 180
        shuffleFeatures = point_nodes[0][:, shuffleIndex, :]  # (1,2708,1433) -- (1, 180, 128)
        logits, h_1, h_2, c = self.dgi(point_nodes[0], shuffleFeatures, point_similarities[0], sparse, None, None, None) #  (1,5416), (1, 180, 32) = (1,2708,1433) (1,2708,1433) (2708,2708)

        # 域特定对比损失
        lbl_1 = torch.ones(self.batch_size, nb_nodes)  # (1,2708) -- (1, 180)
        lbl_2 = torch.zeros(self.batch_size, nb_nodes)  # (1,2708) -- (1, 180)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(dev)  # (1,5416) 111111111111 0000000000000
        BCEWithLogitsLoss = nn.BCEWithLogitsLoss()  # with sigmoid to 0~1; BCE Loss need nn.sigmoid(input);
        loss = BCEWithLogitsLoss(logits , lbl) # scalar

        list1 = [] # point_nodes 类型是list，里边装的tensor
        # list1.append(h_1.transpose(1, 2)) # 不能像java一样把这句话写到返回的那个里 (1, 180, 32) -> (1, 32, 180)
        list1.append(h_1)
        list2 = []
        # list2.append(h_2.transpose(1, 2)) # (1, 180, 32) -> (1, 32, 180)
        list2.append(h_2)
        list = []
        list.append(c) # (1,512)

        return loss, list1, list2, list # (1, 32, 180)

    # def forward(self, point_nodes, point_similarities, sparse = False):
    #     # (2708, 1433) --  (1, 180, 128)  --> (180,128) , 后期需要改，万一batch_size不是1，就不能这么用
    #     # 不要传给下一个变量，传给本身！
    #     point_nodes[0] = torch.FloatTensor(process.preprocess_features(point_nodes[0].squeeze().cpu().detach().numpy())).to(0).unsqueeze(0)
    #     # features = process.preprocess_features(point_nodes[0].squeeze().cpu().detach().numpy())
    #     nb_nodes = point_nodes[0].shape[1]  # 2708 -- 180
    #
    #     adj = point_similarities[0].squeeze().cpu().detach().numpy() # 同样后期需要改，(1, 180, 180)  --> (180,180)
    #     adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    #     adj = (adj + sp.eye(adj.shape[0])).todense()
    #     point_similarities[0] = torch.FloatTensor(adj).to(0).unsqueeze(0)
    #     # features = torch.FloatTensor(features[np.newaxis]).to(0)  # (1,2708,1433) -- (1, 180, 128) 不要用numpy的方法!
    #     # adj = torch.FloatTensor(adj[np.newaxis]).to(0)
    #     shuffleIndex = np.random.permutation(nb_nodes) # 2708 disorder indexes -- 180
    #     shuffleFeatures = point_nodes[0][:, shuffleIndex, :]  # (1,2708,1433) -- (1, 180, 128)
    #     logits = self.dgi(point_nodes[0], shuffleFeatures, point_similarities[0], sparse, None, None, None) #  (1,5416) = (1,2708,1433) (1,2708,1433) (2708,2708)
    #     lbl_1 = torch.ones(self.batch_size, nb_nodes)  # (1,2708) -- (1, 180)
    #     lbl_2 = torch.zeros(self.batch_size, nb_nodes)  # (1,2708) -- (1, 180)
    #     lbl = torch.cat((lbl_1, lbl_2), 1).to(0)  # (1,5416) 111111111111 0000000000000
    #     BCEWithLogitsLoss = nn.BCEWithLogitsLoss()  # with sigmoid to 0~1; BCE Loss need nn.sigmoid(input);
    #     loss = BCEWithLogitsLoss(logits , lbl) # scalar
    #     return loss


    # 之前的
    # def forward(self, point_nodes, point_similarities, sparse = False):
    #     features = process.preprocess_features(point_nodes[0].squeeze().cpu().detach().numpy()) # (2708, 1433) --  (1, 180, 128)  --> (180,128) , 后期需要改，万一batch_size不是1，就不能这么用
    #     nb_nodes = features.shape[0]  # 2708 -- 180
    #     adj = point_similarities[0].squeeze().cpu().detach().numpy() # 同样后期需要改，(1, 180, 180)  --> (180,180)
    #     adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    #     if sparse: # 默认不是稀疏矩阵
    #         sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)  # (2708,2708) -- (180, 180)
    #         sp_adj.to(0)
    #     else:
    #         adj = (adj + sp.eye(adj.shape[0])).todense()
    #     features = torch.FloatTensor(features[np.newaxis]).to(0)  # (1,2708,1433) -- (1, 180, 128)
    #     if not sparse:
    #         adj = torch.FloatTensor(adj[np.newaxis]).to(0)
    #     shuffleIndex = np.random.permutation(nb_nodes) # 2708 disorder indexes -- 180
    #     shuffleFeatures = features[:, shuffleIndex, :]  # (1,2708,1433) -- (1, 180, 128)
    #     logits = self.dgi(features, shuffleFeatures, sp_adj if sparse else adj, sparse, None, None, None) #  (1,5416) = (1,2708,1433) (1,2708,1433) (2708,2708)
    #     lbl_1 = torch.ones(self.batch_size, nb_nodes)  # (1,2708) -- (1, 180)
    #     lbl_2 = torch.zeros(self.batch_size, nb_nodes)  # (1,2708) -- (1, 180)
    #     lbl = torch.cat((lbl_1, lbl_2), 1).to(0)  # (1,5416) 111111111111 0000000000000
    #     BCEWithLogitsLoss = nn.BCEWithLogitsLoss()  # with sigmoid to 0~1; BCE Loss need nn.sigmoid(input);
    #     loss = BCEWithLogitsLoss(logits , lbl) # scalar
    #     return loss











