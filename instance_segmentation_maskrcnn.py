import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging
import math
from lightning.pytorch.utilities import grad_norm

from dataset.instance_segmentation_dataset import LoFiInstanceDataset

from transformers import ViTConfig, ViTModel
from model.mask_rcnn.mask_rcnn import MaskRCNN

from config import cfg

# def custom_collate_fn(batch):
#     return tuple(zip(*batch))


class LoFiViTMaskRCNN(pl.LightningModule):
    def __init__(self, cfg):
        super(LoFiViTMaskRCNN, self).__init__()

        self.cfg = cfg
        
        # use a ViT small model for the backbone from huggingface transformers
        self.vit_config = ViTConfig(num_channels=cfg.DATA.IMAGE.CHANNELS, image_size=cfg.DATA.IMAGE.SIZE)
        self.vit_model = ViTModel(self.vit_config)

        self.maskrcnn_model = MaskRCNN(backbone=self.vit_model, num_classes=1, in_channels=768)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.MODEL.IS.TRAINING.LEARNING_RATE, weight_decay=cfg.MODEL.IS.TRAINING.WEIGHT_DECAY)

    def configure_optimizers(self):
        return self.optimizer
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def forward(self, images, targets):
        result = self.maskrcnn_model(images, targets)
        return result
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = images.squeeze(0)
        targets = {k: v.squeeze(0) for k, v in targets.items()}

        if targets["boxes"].shape[0] == 0:
            return
        
        result = self(images, targets)
        
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return result
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch

        result = self(images, targets)

        # self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return result


if __name__ == "__main__":
    
    cfg.freeze()
    
    pl.seed_everything(cfg.MODEL.SEED_VALUE, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    torch.set_float32_matmul_precision('medium')
    
    # Initialize the dataset and dataloaders
    train_dataset = LoFiInstanceDataset(dataset_path=cfg.DATA.TRAIN_PATH)
    val_dataset = LoFiInstanceDataset(dataset_path=cfg.DATA.VALIDATION_PATH)
    
    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.MODEL.IS.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.MODEL.IS.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=cfg.MODEL.IS.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.MODEL.IS.VALIDATION.NUM_WORKERS, persistent_workers=True)

    print("Train dataset length: ", len(train_dataset))
    print("Validation dataset length: ", len(val_dataset))

    print("Train dataloader length: ", len(train_dataloader))
    print("Validation dataloader length: ", len(val_dataloader))

    model = LoFiViTMaskRCNN(cfg)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        strategy="auto",
        max_epochs=cfg.MODEL.IS.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.MODEL.IS.TRAINING.SWA_LRS)], 
        default_root_dir="/data/hkerner/NASA-MLCommons/lofi/logs/instance_segmentation_vit_maskrcnn"
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg.MODEL.IS.CHECKPOINT_PATH)