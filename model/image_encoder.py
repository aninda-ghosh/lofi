from torch import nn
from transformers import SwinModel, SwinConfig
import math
import torch

class ImageEncoder(nn.Module):
    def __init__(self, image_size, num_channels, freeze=False):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.freeze = freeze

        self.swinConfig = SwinConfig(image_size=self.image_size, num_channels=self.num_channels)
        self.swinModel = SwinModel(self.swinConfig, add_pooling_layer=False, use_mask_token=True)

        if self.freeze:
            for param in self.swinModel.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, bool_masked_pos=None):
        outputs = self.swinModel(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)

        return outputs
        