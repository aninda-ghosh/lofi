# from torchvision.ops import MultiScaleRoIAlign
# from torchvision.models.detection.anchor_utils import AnchorGenerator
# from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
# from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
# from torchvision.models.detection.roi_heads import RoIHeads

import torch.nn as nn
import torch


class MaskDecoder(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(MaskDecoder, self).__init__()
        
        # Upsampling layers
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Final convolutional layer
        self.conv_layer = nn.Conv2d(in_channels=6, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        
        # Activation function
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Upsampling
        x = self.upsampler(x)
        
        # Final convolution and activation
        x = self.conv_layer(x)
        x = self.activation(x)

        #unsqueeze the mask_logits in domension 1
        x = x.squeeze(1)
        
        return x
        