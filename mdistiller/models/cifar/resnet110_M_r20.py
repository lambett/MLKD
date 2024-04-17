import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torchvision import models
# from resnet_56 import *
import torch.nn.functional as F
# from block_base import block_base
# from block_mid import block_mid_1, block_mid_2
# from block_group import block_group
import numpy as np
from mdistiller.models.cifar.resnet import resnet20, resnet32, resnet44
# from models import *
import scipy.io as scio
import pandas as pd
import copy

# load_path = '/home/xuk/桌面/git_code/pytorch-cifar-master/checkpoint/vgg16_bn-6c64b313.pth'
# pretrained_dict = torch.load(load_path)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNRelu, self).__init__()

        self.conbnrelu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(0.1, inplace=True)
        )

        for m in self.conbnrelu.children():
            m.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.conbnrelu(x)

        return x





class classifier(nn.Module):
    def __init__(self, in_features, class_num):
        super(classifier, self).__init__()

        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(in_features=in_features, out_features=class_num),
        )

        for m in self.classifier.children():
            m.apply(weights_init_classifier)

    def forward(self, x):
        x = self.classifier(x)

        return x


class res110_M_r20(nn.Module):
    def __init__(self, num_classes=100):
        super(res110_M_r20, self).__init__()
        self.class_num = num_classes
        self.proj_head_in_features = 192
        self.base = resnet20()
        # self.base.layer1 = nn.Sequential()
        # self.base.layer2 = nn.Sequential()
        # self.base.layer3 = nn.Sequential()
        # self.base.linear = nn.Sequential()
        # self.outblock = resnet20()
        # self.outblock.conv1 = nn.Sequential()
        # self.outblock.bn1 = nn.Sequential()
        # self.outblock.layer1 = nn.Sequential()
        # self.outblock.layer2 = nn.Sequential()
        # self.outblock.fc = nn.Linear(64, 100)

        self.res_1 = resnet20()
        self.res_1.conv1 = nn.Sequential()
        self.res_1.bn1 = nn.Sequential()
        self.res_1.layer1 = nn.Sequential()
        self.res_1.layer2 = nn.Sequential()
        self.res_1.fc = nn.Sequential()

        self.res_2 = resnet20()
        self.res_2.conv1 = nn.Sequential()
        self.res_2.bn1 = nn.Sequential()
        self.res_2.layer1 = nn.Sequential()
        self.res_2.layer2 = nn.Sequential()
        self.res_2.fc = nn.Sequential()

        self.res_3 = resnet20()
        self.res_3.conv1 = nn.Sequential()
        self.res_3.bn1 = nn.Sequential()
        self.res_3.layer1 = nn.Sequential()
        self.res_3.layer2 = nn.Sequential()
        self.res_3.fc = nn.Sequential()

        self.block_group_1_classifier = nn.Sequential(
            classifier(in_features=64, class_num=33),
        )
        self.block_group_2_classifier = nn.Sequential(
            classifier(in_features=64, class_num=33),
        )
        self.block_group_3_classifier = nn.Sequential(
            classifier(in_features=64, class_num=34),
        )

    def set_diff_level(self, dif_labels, labels):
        class_easy = []
        class_mid = []
        class_hard = []

        indices_easy = []
        indices_mid = []
        indices_hard = []

        class_indices = {}
        labels = labels.cpu().numpy()
        for i, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        class_diff_splits = {}
        flag_l = []
        for label, indices in class_indices.items():
            easy = [i for i in indices if dif_labels[i] == 0]
            medium = [i for i in indices if dif_labels[i] == 1]
            hard = [i for i in indices if dif_labels[i] == 2]
            e_num = len(easy)
            m_num = len(medium)
            h_num = len(hard)
            flag = h_num * 1 + m_num * 0.5 + e_num * 0.1
            class_diff_splits[label] = flag
            flag_l.append(flag)
        flag_l.sort()

        for label, indices in class_indices.items():
            if class_diff_splits[label] < flag_l[int(len(flag_l) / 3)]:
                class_easy.append(label)
                for i in indices:
                    indices_easy.append(i)
            elif class_diff_splits[label] < flag_l[int(2 * len(flag_l) / 3)]:
                class_mid.append(label)
                for i in indices:
                    indices_mid.append(i)
            else:
                class_hard.append(label)
                for i in indices:
                    indices_hard.append(i)

        self.class_easy = torch.tensor(np.array(class_easy).astype(int), dtype=torch.int64).cuda()
        self.class_mid = torch.tensor(np.array(class_mid).astype(int), dtype=torch.int64).cuda()
        self.class_hard = torch.tensor(np.array(class_hard).astype(int), dtype=torch.int64).cuda()

    def forward(self, x, is_feat=False):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x, _ = self.base.layer1(x)
        x, _ = self.base.layer2(x)

        x_1, _ = self.res_1.layer3(x)
        cov_1 = x_1.clone().detach()
        x_1 = F.avg_pool2d(x_1, x_1.size()[3])
        x_1 = x_1.view(x_1.size(0), -1)
        x_1_out = self.block_group_1_classifier(x_1)

        x_2, _ = self.res_2.layer3(x)
        cov_2 = x_2.clone().detach()
        x_2 = F.avg_pool2d(x_2, x_2.size()[3])
        x_2 = x_2.view(x_2.size(0), -1)
        x_2_out = self.block_group_2_classifier(x_2)

        x_3, _ = self.res_3.layer3(x)
        cov_3 = x_3.clone().detach()
        x_3 = F.avg_pool2d(x_3, x_3.size()[3])
        x_3 = x_3.view(x_3.size(0), -1)
        x_3_out = self.block_group_3_classifier(x_3)

        preds = []

        class_1 = self.class_easy
        class_2 = self.class_mid
        class_3 = self.class_hard

        num_1 = 0
        num_2 = 0
        num_3 = 0

        for i in range(100):
            if (i in class_1):
                preds.append(x_1_out[:, num_1].reshape([-1, 1]))
                num_1 += 1

            elif (i in class_2):
                preds.append(x_2_out[:, num_2].reshape([-1, 1]))
                num_2 += 1

            elif (i in class_3):
                preds.append(x_3_out[:, num_3].reshape([-1, 1]))
                num_3 += 1

        # for i in range(100):
        #     if (i in class_1):
        #         preds.append(x_1_out[:, i].reshape([-1, 1]))
        #
        #     elif (i in class_2):
        #         preds.append(x_2_out[:, i].reshape([-1, 1]))
        #
        #     elif (i in class_3):
        #         preds.append(x_3_out[:, i].reshape([-1, 1]))

        out_multi = torch.cat(preds, dim=1)

        feats = torch.cat([x_1, x_2, x_3], dim=1)

        covout = torch.cat([cov_1, cov_2, cov_3], dim=1)

        # out_sigle, _ = self.base.layer3(x)
        # out = F.avg_pool2d(out_sigle, out_sigle.size()[3])
        # out = out.view(out.size(0), -1)
        # out_sigle = self.base.fc(out)

        return feats, out_multi, covout


if __name__ == '__main__':
    model = res110_M_r20()
    # print(model)
    x = torch.randn([64, 3, 32, 32])
    # x_1, x_2, x_3_1, x_3_2, x_3_3, x_4 = model(x)
    # print(x_1.shape, x_2.shape, x_3_1.shape, x_3_2.shape, x_3_3.shape, x_4.shape)
    x, y = model(x)
    print(x.shape)
    print(y.shape)
    # g = make_dot(y)
    # g.render('models_small_full', view=False)

