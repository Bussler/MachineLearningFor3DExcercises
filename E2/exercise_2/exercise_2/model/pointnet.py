import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        # T Add layers: Convolutional k->64, 64->128, 128->1024 with corresponding batch norms and ReLU
        # T Add layers: Linear 1024->512, 512->256, 256->k^2 with corresponding batch norms and ReLU

        self.convolutions = nn.Sequential(
            nn.Conv1d(k,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.linearLayers = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256,k**2)
        )

        self.register_buffer('identity', torch.from_numpy(np.eye(k).flatten().astype(np.float32)).view(1, k ** 2))
        self.k = k

    def forward(self, x):
        b = x.shape[0]

        # T Pass input through layers, applying the same max operation as in PointNetEncoder
        # T No batch norm and relu after the last Linear layer

        # BigM: Input x.shape = (batch_size, 3, 1024) or (batch_size, 64, 1024)
        x = self.convolutions(x) # BigM: don't need to flatten x, since first conv layer takes the 3 vals x, y, z as input or 64 per point features after feature transform

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.linearLayers(x)

        # Adding the identity to constrain the feature transformation matrix to be close to orthogonal matrix
        identity = self.identity.repeat(b, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, return_point_features=False):
        super().__init__()

        # T Define convolution layers, batch norm layers, and ReLU

        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.input_transform_net = TNet(k=3)
        self.feature_transform_net = TNet(k=64)

        self.return_point_features = return_point_features

    def forward(self, x):
        num_points = x.shape[2]

        input_transform = self.input_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), input_transform).transpose(2, 1)

        # T First layer: 3->64
        x = F.relu(self.bn1(self.conv1(x)))

        feature_transform = self.feature_transform_net(x)
        x = torch.bmm(x.transpose(2, 1), feature_transform).transpose(2, 1)
        point_features = x

        # T Layers 2 and 3: 64->128, 128->1024

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # TODO no relu correct here, since last layer?

        # This is the symmetric max operation
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.return_point_features:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_features], dim=1)
        else:
            return x


class PointNetClassification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = PointNetEncoder(return_point_features=False)
        # T Add Linear layers, batch norms, dropout with p=0.3, and ReLU
        # Batch Norms and ReLUs are used after all but the last layer
        # Dropout is used only directly after the second Linear layer
        # The last Linear layer reduces the number of feature channels to num_classes (=k in the architecture visualization)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = self.encoder(x)
        # T Pass output of encoder through your layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # second layer, dropout before batch norm
        x = self.fc3(x) # BigM: put LogSoftMax on top?

        return x


class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = PointNetEncoder(return_point_features=True)
        # TODO: Define convolutions, batch norms, and ReLU

        self.conv1 = nn.Conv1d(1088,512,1)
        self.conv2 = nn.Conv1d(512,256,1)
        self.conv3 = nn.Conv1d(256,128,1)
        self.conv4 = nn.Conv1d(128,num_classes,1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.encoder(x)
        # T Pass x through all layers, no batch norm or ReLU after the last conv layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1).contiguous()
        return x
