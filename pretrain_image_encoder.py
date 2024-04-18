import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging
import math

from dataset.image_pretrain_dataset import PretrainDataset
from model.image_encoder import ImageEncoder

from config import cfg

class PretrainImageEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super(PretrainImageEncoder, self).__init__()

        self.cfg = cfg
        self.image_encoder = ImageEncoder(image_size=cfg.DATA.IMAGE.SIZE, num_channels=cfg.DATA.IMAGE.CHANNELS)

        num_features = int(self.image_encoder.swinConfig.embed_dim * 2 ** (self.image_encoder.swinConfig.num_layers - 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=self.image_encoder.swinConfig.encoder_stride**2 * self.image_encoder.swinConfig.num_channels, kernel_size=1
            ),
            nn.PixelShuffle(self.image_encoder.swinConfig.encoder_stride)
        )

        self.num_patches = (self.image_encoder.swinModel.config.image_size // self.image_encoder.swinModel.config.patch_size) ** 2
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.MODEL.SSL.TRAINING.LEARNING_RATE, weight_decay=cfg.MODEL.SSL.TRAINING.WEIGHT_DECAY)

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, image, batch_size):
        mask_positions = torch.randint(low=0, high=2, size=(batch_size, self.num_patches)).bool()
        mask_positions = mask_positions.to(self.device)
        
        image_embedding = self.image_encoder(pixel_values=image, bool_masked_pos=mask_positions)
        
        sequence_output = image_embedding[0]
        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output.transpose(1, 2)
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if mask_positions is not None:
            size = self.image_encoder.swinConfig.image_size // self.image_encoder.swinConfig.patch_size
            mask_positions = mask_positions.reshape(-1, size, size)
            mask = (
                mask_positions.repeat_interleave(self.image_encoder.swinConfig.patch_size, 1)
                .repeat_interleave(self.image_encoder.swinConfig.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(image, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.image_encoder.swinConfig.num_channels
        
        output = (reconstructed_pixel_values,) + image_embedding[2:]

        return ((masked_im_loss,) + output) if masked_im_loss is not None else output
    
    def training_step(self, batch, batch_idx):
        image = batch
        
        loss, _ = self(image, batch.shape[0])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch
        
        loss, _ = self(image, batch.shape[0])

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


if __name__ == "__main__":
    pl.seed_everything(cfg.MODEL.SEED_VALUE, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    

    torch.set_float32_matmul_precision('medium')

    # Initialize the dataset and dataloaders
    train_dataset = PretrainDataset(dataset_path=cfg.DATA.TRAIN_PATH)
    validation_dataset = PretrainDataset(dataset_path=cfg.DATA.VALIDATION_PATH)

    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.MODEL.SSL.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.MODEL.SSL.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(validation_dataset, pin_memory=True, batch_size=cfg.MODEL.SSL.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.MODEL.SSL.VALIDATION.NUM_WORKERS, persistent_workers=True)

    model = PretrainImageEncoder(cfg)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        strategy="auto",
        max_epochs=cfg.MODEL.SSL.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.MODEL.SSL.TRAINING.SWA_LRS)], 
        default_root_dir="/data/hkerner/NASA-MLCommons/lofi/logs/ssl"
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg.MODEL.SSL.CHECKPOINT_PATH)