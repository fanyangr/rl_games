import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class PointNet(nn.Module):
    def __init__(self, k=40, normal_channel=False, center_input=False, history_length=1):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=channel, center_input=center_input)
        self.fc1 = nn.Linear(256 * history_length, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.history_length = history_length # it can take pointcloud of multiple frames

    def forward(self, x):
        # x has shape [B, 3, N * history_length]
        x = x.reshape(x.size(0), x.size(1), x.size(2) // self.history_length, self.history_length)
        pointcloud_features = []
        for i in range(self.history_length):
            pointcloud_feature, trans, trans_feat = self.feat(x[:, :, :, i])
            pointcloud_features.append(pointcloud_feature)
        x = torch.cat(pointcloud_features, dim=-1)
        # x, trans, trans_feat = self.feat(x)        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
# class PointNet(nn.Module):
#     def __init__(self, k=40, normal_channel=False):
#         super(PointNet, self).__init__()
#         if normal_channel:
#             channel = 6
#         else:
#             channel = 3
#         self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=channel)
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, k)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x, trans, trans_feat = self.feat(x)        
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = self.fc2(x)
#         return x
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3, center_input=False):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 16, 1)
        self.conv2 = torch.nn.Conv1d(16, 32, 1)
        self.conv3 = torch.nn.Conv1d(32, 256, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.center_input = center_input
        if self.feature_transform:
            self.fstn = STNkd(k=16)

    def forward(self, x):
        B, D, N = x.size()  # NOTE: the input shape is different, it is B, D, N

        if self.center_input:
            if D >= 3:
                centroid = x[:, :3, :].mean(dim=2, keepdim=True)
                if D > 3:
                    x = torch.cat([x[:, :3, :] - centroid, x[:, 3:, :]], dim=1)
                else:
                    x = x[:, :3, :] - centroid

        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 256, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 16, 1)
        self.conv2 = torch.nn.Conv1d(16, 32, 1)
        self.conv3 = torch.nn.Conv1d(32, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
