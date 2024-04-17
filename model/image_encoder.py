from torch import nn
from transformers import SwinModel, SwinConfig
import math

class ImageEncoder(nn.Module):
    def __init__(self, image_size, num_channels, pretrain=True):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.pretrain = pretrain

        self.swinConfig = SwinConfig(image_size=self.image_size, num_channels=self.num_channels)
        self.swinModel = SwinModel(self.swinConfig, add_pooling_layer=False, use_mask_token=True)

        if self.pretrain:
            num_features = int(self.swinConfig.embed_dim * 2 ** (self.swinConfig.num_layers - 1))
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_features, out_channels=self.swinConfig.encoder_stride**2 * self.swinConfig.num_channels, kernel_size=1
                ),
                nn.PixelShuffle(self.swinConfig.encoder_stride)
            )
        else:
            for param in self.swinModel.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, bool_masked_pos):
        outputs = self.swinModel(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)

        if self.pretrain:
            sequence_output = outputs[0]
            # Reshape to (batch_size, num_channels, height, width)
            sequence_output = sequence_output.transpose(1, 2)
            batch_size, num_channels, sequence_length = sequence_output.shape
            height = width = math.floor(sequence_length**0.5)
            sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)

            # Reconstruct pixel values
            reconstructed_pixel_values = self.decoder(sequence_output)

            masked_im_loss = None
            if bool_masked_pos is not None:
                size = self.swinConfig.image_size // self.swinConfig.patch_size
                bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
                mask = (
                    bool_masked_pos.repeat_interleave(self.swinConfig.patch_size, 1)
                    .repeat_interleave(self.swinConfig.patch_size, 2)
                    .unsqueeze(1)
                    .contiguous()
                )
                reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
                masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.swinConfig.num_channels

            output = (reconstructed_pixel_values,) + outputs[2:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output
        else:
            return outputs
        