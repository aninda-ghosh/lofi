import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging

from dataset.dataset import LoFiDataset
from model.image_encoder import ImageEncoder

from config import cfg

class PretrainImageEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super(PretrainImageEncoder, self).__init__()

        self.cfg = cfg
        self.model = ImageEncoder(image_size=cfg.image_size, num_channels=cfg.num_channels, pretrain=True)
        self.num_patches = (self.model.config.image_size // self.model.config.patch_size) ** 2
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.TRAINING.LEARNING_RATE, weight_decay=cfg.TRAINING.WEIGHT_DECAY)


    def forward(self, image):
        mask_positions = torch.randint(low=0, high=2, size=(1, self.num_patches)).bool()
        outputs = self.model(pixel_values= image, bool_masked_pos=mask_positions)
        loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        return loss, reconstructed_pixel_values
    
    def training_step(self, batch, batch_idx):
        image = batch
        
        loss, _ = self.model(image)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch
        
        loss, _ = self.model(image)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


if __name__ == "__main__":
    pl.seed_everything(cfg.MODEL.SEED_VALUE, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    
    torch.set_float32_matmul_precision('medium')

    # Initialize the dataset and dataloaders
    dataset = LoFiDataset(dataset_path=cfg.DATA.TRAIN_DATASET_PATH)
    
    train_size = int(cfg.TRAINING.TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])


    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=cfg.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.VALIDATION.NUM_WORKERS, persistent_workers=True)

    model = PretrainImageEncoder(cfg)
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        strategy="auto",
        max_epochs=cfg.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.TRAINING.SWA_LRS)]
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)