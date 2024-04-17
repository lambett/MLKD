import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class wrapper_sp(nn.Module):

    def __init__(self, module, feat_dim, class_num=100, radius=10.0):

        super(wrapper_sp, self).__init__()
        self.radius = radius
        self.backbone = module
        in_features = list(module.children())[-1].in_features
        self.proj_head = nn.Sequential(
            SP_layer(in_features, feat_dim, self.radius),
            SP_layer(feat_dim, feat_dim, self.radius)
            )
        self.l2norm = Normalize(2)
        self.classifier = nn.Linear(feat_dim, class_num)
        # self.classifier = SLR_layer(feat_dim, class_num)
        self.weight_check = None
        self.iter_num = 0

    def forward(self, x, bb_grad=True):
        if self.training:
            self.iter_num += 1
        with torch.no_grad():
            out, feats = self.backbone(x)
        feat = feats["pooled_feat"]
        if not bb_grad:
            feat = feat.detach()

        proj_x = self.proj_head(feat)
        proj_x = self.l2norm(proj_x)
        proj_x = proj_x * 16.0
        proj_logit = self.classifier(proj_x)
        return feat, out, proj_x, proj_logit


class wrapper_sp_s(nn.Module):

    def __init__(self, module, feat_dim, radius=10.0):
        super(wrapper_sp_s, self).__init__()

        self.radius = radius
        self.backbone = module
        # in_features = 64
        in_features = module.proj_head_in_features
        self.proj_head = nn.Sequential(
            SP_layer(in_features, feat_dim, self.radius),
            SP_layer(feat_dim, feat_dim, self.radius)
        )
        self.l2norm = Normalize(2)
        print('..........................................................................................')

    def set_diff_level(self, dif_labels, labels):
        self.backbone.set_diff_level(dif_labels, labels)

    def forward(self, x, bb_grad=True):
        feat, out, cov = self.backbone(x, is_feat=True)
        # feat = feat[-1].view(feat[-1].size(0), -1)
        feat = feat.view(feat.size(0), -1)
        if not bb_grad:
            feat = feat.detach()
        proj_x = self.proj_head(feat)
        proj_x = self.l2norm(proj_x)
        proj_x = proj_x * 16.0
        return feat, out, proj_x, cov


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r*torch.tanh(self.bias))
        output = cosine
        return output


class SP_layer(nn.Module):
    def __init__(self, in_feature,out_feature,radius):
        super(SP_layer, self).__init__()
        self.radius = radius
        self.linear = nn.Linear(in_feature-1,out_feature-1)
        self.dropout = nn.Dropout(0.5)
        self.apply(init_weights)

    def forward(self, x):
        v = self.log(x, r=self.radius)
        v = self.linear(v)
        v = self.dropout(v)
        x = self.exp(v, r=self.radius)
        x = self.srelu(x, r=self.radius)
        return x

    def exp(self, v, o=None, r=1.0):
        if v.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            o = torch.cat([torch.zeros(1, v.size(1)), r * torch.Tensor([[1]])], dim=1).to(device)
        theta = torch.norm(v, dim=1, keepdim=True) / r
        v = torch.cat([v, torch.zeros(v.size(0), 1).to(device)], dim=1)
        return torch.cos(theta) * o + torch.sin(theta) * F.normalize(v, dim=1) * r

    def log(self,x, o=None, r=1.0):
        if x.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            o = torch.cat([torch.zeros(1, x.size(1) - 1), r * torch.Tensor([[1]])], dim=1).to(device)
        c = F.cosine_similarity(x, o, dim=1).view(-1, 1)
        theta = torch.acos(self.shrink(c))
        v = F.normalize(x - c * o, dim=1)[:, :-1]
        return r * theta * v

    def shrink(self,x, epsilon=1e-4):
        x[torch.abs(x) > (1 - epsilon)] = x[torch.abs(x) > (1 - epsilon)] * (1 - epsilon)
        return x

    def srelu(self,x, r=1.0):
        return r * F.normalize(F.relu(x), dim=1)


