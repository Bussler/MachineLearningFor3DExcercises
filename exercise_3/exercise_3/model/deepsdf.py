import torch.nn as nn
import torch


# M: Class to define the nws of the sequence of weight_norm networks
class WeightNormNW(nn.Module):
    def __init__(self, inDim, outDim, dropout_prob):
        self.Network = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(inDim, outDim)), #M TODO add batch norm layer here?
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )

    def forward(self, x):
        x = self.Network(x)
        return x


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # T: Define model
        self.WNW1 = WeightNormNW(latent_size+3,512,dropout_prob)
        self.WNW2 = WeightNormNW(512,512,dropout_prob)
        self.WNW3 = WeightNormNW(512,512,dropout_prob)
        self.WNW4 = WeightNormNW(512,512-(latent_size+3),dropout_prob)
        self.WNW5 = WeightNormNW(512,512,dropout_prob)
        self.WNW6 = WeightNormNW(512,512,dropout_prob)
        self.WNW7 = WeightNormNW(512,512,dropout_prob)
        self.WNW8 = WeightNormNW(512,512,dropout_prob)

        self.ll = nn.Linear(512, 1)

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        return x
