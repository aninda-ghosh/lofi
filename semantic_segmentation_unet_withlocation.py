import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging
import math
from lightning.pytorch.utilities import grad_norm

from dataset.semantic_segmentation_dataset import LoFiSemanticDataset

from model.custom_unet import UNetEncoder, UNetDecoder, UNetBottleneck
from model.location_encoder import LocationEncoder

from solver.losses import DiceLoss

from config import cfg



class LoFiUNet(pl.LightningModule):
    def __init__(self, cfg):
        super(LoFiUNet, self).__init__()

        self.cfg = cfg
        self.encoder = UNetEncoder(in_channels=cfg.DATA.IMAGE.CHANNELS)
        self.bottleneck = UNetBottleneck(in_channels=512+1)  #! 512 is the number of output channels from the encoder, 1 is the number of channels from the location encoder
        self.decoder = UNetDecoder(out_channels=1)

        self.location_encoder = LocationEncoder(embedding_size=1024)

        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.MODEL.SSUL.TRAINING.LEARNING_RATE, weight_decay=cfg.MODEL.SSUL.TRAINING.WEIGHT_DECAY)

    def configure_optimizers(self):
        return self.optimizer
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def forward(self, image, gps):
        s1, s2, s3, s4, p4 = self.encoder(image)
        
        # Get the GPS features from the location encoder
        gps_embedding = self.location_encoder(gps)
        gps_embedding = F.normalize(gps_embedding, dim=1)
        gps_embedding = gps_embedding.reshape(image.shape[0], 1, 32, 32)
        
        p4 = torch.cat([p4, gps_embedding], axis=1)

        b = self.bottleneck(p4, gps)
        mask_logits = self.decoder(b, s1, s2, s3, s4)

        return mask_logits
    
    def training_step(self, batch, batch_idx):
        image, gps, target_masks = batch
        target_masks = target_masks.squeeze(1)


        mask_logits = self(image, gps).squeeze(1)

        _bce_loss = self.bce_loss(mask_logits, target_masks)
        mask_logits = torch.where(mask_logits > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        _dice_loss = self.dice_loss(mask_logits, target_masks)

        loss = _bce_loss + _dice_loss

        loss = loss.mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        image, gps, target_masks = batch
        target_masks = target_masks.squeeze(1)


        mask_logits = self(image, gps).squeeze(1)

        _bce_loss = self.bce_loss(mask_logits, target_masks)
        mask_logits = torch.where(mask_logits > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        _dice_loss = self.dice_loss(mask_logits, target_masks)

        loss = _bce_loss + _dice_loss

        loss = loss.mean()

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


if __name__ == "__main__":
    
    cfg.freeze()
    
    pl.seed_everything(cfg.MODEL.SEED_VALUE, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    torch.set_float32_matmul_precision('medium')
    
    # Initialize the dataset and dataloaders
    train_dataset = LoFiSemanticDataset(dataset_path=cfg.DATA.TRAIN_PATH)
    val_dataset = LoFiSemanticDataset(dataset_path=cfg.DATA.VALIDATION_PATH)
    
    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.MODEL.SSUL.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.MODEL.SSUL.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=cfg.MODEL.SSUL.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.MODEL.SSUL.VALIDATION.NUM_WORKERS, persistent_workers=True)

    print("Train dataset length: ", len(train_dataset))
    print("Validation dataset length: ", len(val_dataset))

    print("Train dataloader length: ", len(train_dataloader))
    print("Validation dataloader length: ", len(val_dataloader))

    model = LoFiUNet(cfg)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=[1], 
        strategy="auto",
        max_epochs=cfg.MODEL.SSUL.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.MODEL.SSUL.TRAINING.SWA_LRS)], 
        default_root_dir="/data/hkerner/NASA-MLCommons/lofi/logs/semantic_segmentation_unet_withlocation"
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg.MODEL.SSUL.CHECKPOINT_PATH)