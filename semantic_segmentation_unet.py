import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging
import math
from lightning.pytorch.utilities import grad_norm

from dataset.semantic_segmentation_dataset import LoFiSemanticDataset

from model.unet import UNet

from solver.losses import DiceLoss

from config import cfg



class LoFiUNet(pl.LightningModule):
    def __init__(self, cfg):
        super(LoFiUNet, self).__init__()

        self.cfg = cfg
        # UNET model from pytorch
        self.unet_model = UNet(in_channels=cfg.DATA.IMAGE.CHANNELS, out_channels=1)

        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.MODEL.SSU.TRAINING.LEARNING_RATE, weight_decay=cfg.MODEL.SSU.TRAINING.WEIGHT_DECAY)

    def configure_optimizers(self):
        return self.optimizer
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def forward(self, image, gps, batch_size):
        mask_logits = self.unet_model(image)

        return mask_logits
    
    def training_step(self, batch, batch_idx):
        image, gps, target_masks = batch
        target_masks = target_masks.squeeze(1)


        mask_logits = self(image, gps, image.shape[0]).squeeze(1)

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


        mask_logits = self(image, gps, image.shape[0]).squeeze(1)

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
    
    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.MODEL.SSU.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.MODEL.SSU.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=cfg.MODEL.SSU.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.MODEL.SSU.VALIDATION.NUM_WORKERS, persistent_workers=True)

    print("Train dataset length: ", len(train_dataset))
    print("Validation dataset length: ", len(val_dataset))

    print("Train dataloader length: ", len(train_dataloader))
    print("Validation dataloader length: ", len(val_dataloader))

    model = LoFiUNet(cfg)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=[0], 
        strategy="auto",
        max_epochs=cfg.MODEL.SSU.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.MODEL.SSU.TRAINING.SWA_LRS)], 
        default_root_dir="/data/hkerner/NASA-MLCommons/lofi/logs/semantic_segmentation_unet"
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg.MODEL.SSU.CHECKPOINT_PATH)