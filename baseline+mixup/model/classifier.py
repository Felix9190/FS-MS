import torch.nn as nn
import torch.nn.functional as F

class C_F_Classifier(nn.Module):
    '''
        in_channels = 160
        hidden_channels = 64
        num_C_cls = 12, 10, 8, 6, 4
        num_F_cls = TAR_CLASS_NUM
    '''
    def __init__(self, in_channels, hidden_channels, num_C_cls, num_F_cls):
        super(C_F_Classifier, self).__init__()
        self.mlp_c = nn.Sequential(nn.Linear(in_features=in_channels,
                                             out_features=hidden_channels),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(in_features=hidden_channels,
                                             out_features=num_C_cls))
        self.mlp_f = nn.Sequential(nn.Linear(in_features=in_channels,
                                             out_features=hidden_channels),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(in_features=hidden_channels,
                                             out_features=num_F_cls))

    def forward(self, x): # (batch_size, 160)
        x_c = F.softmax(self.mlp_c(x), dim=1) # (batch_size, num_C_cls)
        x_f = F.softmax(self.mlp_f(x), dim=1) # (batch_size, num_F_cls)

        return x_c, x_f