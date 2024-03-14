import torch.nn as nn

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, prev_dim = 160, dim = 128, pred_dim = 32):
        """
        prev_dim: feature dimension (default: 160)
        dim: after project, feature dimension (default: 128)
        pred_dim: hidden dimension of the predictor (default: 32)
        """
        super(SimSiam, self).__init__()
        # 【注意】这个参数数量很多，如果有效果且时间太长的话，可以把 prev_dim = 160 先映射成128或者64.
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True), # first layer
                                       nn.Linear(prev_dim, prev_dim, bias=False),
                                       nn.BatchNorm1d(prev_dim),
                                       nn.ReLU(inplace=True), # second layer
                                       nn.Linear(prev_dim, dim, bias=False),
                                       nn.BatchNorm1d(dim, affine=False)) # 会导致weight == none, bias == none

        # build a 2-layer predictor  dim = 128  pred_dim = 32
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True), # hidden layer
                                       nn.Linear(pred_dim, dim)) # output layer

    def forward(self, h1, h2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        # compute features for one view
        z1 = self.projector(h1) # (class * (1 + 19), 128)
        z2 = self.projector(h2) # (class * (1 + 19), 128)
        p1 = self.predictor(z1) # (class * (1 + 19), 128)
        p2 = self.predictor(z2) # (class * (1 + 19), 128)

        return p1, p2, z1.detach(), z2.detach()
