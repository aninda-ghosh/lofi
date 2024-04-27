import torch
import torch.nn as nn

from model.equal_earth_projection import equal_earth_projection
from model.random_fourier_features import GaussianEncoding

class LocationEncoderSingleFourierLayer(nn.Module):
    def __init__(self, sigma, embedding_size):
        super(LocationEncoderSingleFourierLayer, self).__init__()
        self.embedding_size = embedding_size
        self.fourier_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=self.embedding_size//2)
        self.linear1 = nn.Linear(self.embedding_size, 1024)
        self.activ1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1024)
        self.activ2 = nn.ReLU()
        self.linear3 = nn.Linear(1024, 1024)
        self.activ3 = nn.ReLU()
        self.head = nn.Sequential(nn.Linear(1024, self.embedding_size))

    def forward(self, x):
        x = self.fourier_encoding(x)
        x = self.activ1(self.linear1(x))
        x = self.activ2(self.linear2(x))
        x = self.activ3(self.linear3(x))
        x = self.head(x)
        return x
    

class LocationEncoder(nn.Module):
    def __init__(self, embedding_size, sigma=[2**0, 2**4, 2**8], freeze=False):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.embedding_size = embedding_size
        self.num_layers = len(self.sigma)
        self.freeze = freeze

        for i, s in enumerate(self.sigma):
            self.add_module('LocationEncoderLayer' + str(i), LocationEncoderSingleFourierLayer(sigma=s, embedding_size=self.embedding_size))

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, location):
        location = equal_earth_projection(location)
        location_features = torch.zeros(location.shape[0], self.embedding_size).to(location.device)

        for i in range(self.num_layers):
            location_features += self._modules['LocationEncoderLayer' + str(i)](location)
        
        return location_features