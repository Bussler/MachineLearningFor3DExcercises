import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # T: 4 Encoder layers
        self.enConv1 = nn.Conv3d(in_channels= 2, out_channels= self.num_features, kernel_size= 4, stride= 2, padding= 1)
        self.enConv2 = nn.Conv3d(in_channels= self.num_features, out_channels= 2 * self.num_features, kernel_size= 4, stride= 2, padding= 1)
        self.enConv3 = nn.Conv3d(in_channels= 2 * self.num_features, out_channels= 4 * self.num_features, kernel_size= 4, stride= 2, padding= 1)
        self.enConv4 = nn.Conv3d(in_channels= 4 * self.num_features, out_channels= 8 * self.num_features, kernel_size= 4, stride= 1, padding= 0)

        self.enBn2 = nn.BatchNorm3d(2 * self.num_features)
        self.enBn3 = nn.BatchNorm3d(4 * self.num_features)
        self.enBn4 = nn.BatchNorm3d(8 * self.num_features)

        # T: 2 Bottleneck layers

        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features * 8, self.num_features * 8),
            nn.ReLU(),

            nn.Linear(self.num_features * 8, self.num_features * 8),
            nn.ReLU(),
        )

        # T: 4 Decoder layers
        self.deConv1 = nn.ConvTranspose3d(in_channels= self.num_features * 8 * 2, out_channels= self.num_features * 4, kernel_size= 4, stride= 1, padding= 0) #M TODO * 2 bei input correct?
        self.deConv2 = nn.ConvTranspose3d(in_channels= self.num_features * 4 * 2, out_channels= self.num_features * 2, kernel_size= 4, stride= 2, padding= 1)
        self.deConv3 = nn.ConvTranspose3d(in_channels= self.num_features * 2 * 2, out_channels= self.num_features, kernel_size= 4, stride= 2, padding= 1)
        self.deConv4 = nn.ConvTranspose3d(in_channels= self.num_features * 2, out_channels= 1, kernel_size= 4, stride= 2, padding= 1)

        self.deBn1 = nn.BatchNorm3d(self.num_features * 4)
        self.deBn2 = nn.BatchNorm3d(self.num_features * 2)
        self.deBn3 = nn.BatchNorm3d(self.num_features)

    def forward(self, x):
        b = x.shape[0]
        # Encode
        # T: Pass x though encoder while keeping the intermediate outputs for the skip connections

        x_e1 = F.leaky_relu(self.enConv1(x), negative_slope=0.2)
        x_e2 = F.leaky_relu(self.enBn2(self.enConv2(x_e1)), negative_slope=0.2)
        x_e3 = F.leaky_relu(self.enBn3(self.enConv3(x_e2)), negative_slope=0.2)
        x_e4 = F.leaky_relu(self.enBn4(self.enConv4(x_e3)), negative_slope=0.2)

        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # T: Pass x through the decoder, applying the skip connections in the process

        x = torch.cat([x, x_e4], dim= 1) # M concat along dim 1, since dim 0 is batch
        x = F.relu(self.deBn1(self.deConv1(x)))

        x = torch.cat([x, x_e3], dim= 1)
        x = F.relu(self.deBn2(self.deConv2(x)))

        x = torch.cat([x, x_e2], dim= 1)
        x = F.relu(self.deBn3(self.deConv3(x)))

        x = torch.cat([x, x_e1], dim= 1)
        x = F.relu(self.deConv4(x))

        x = torch.squeeze(x, dim=1)
        # T: Log scaling

        x = torch.log(torch.abs(x)+1)

        return x
