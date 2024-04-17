import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging

from dataset.image_pretrain_dataset import LoFiDataset
from model.image_encoder import ImageEncoder

from config import cfg

class PretrainImageEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super(PretrainImageEncoder, self).__init__()

        self.cfg = cfg
        self.model = ImageEncoder(image_size=cfg.DATA.IMAGE.SIZE, num_channels=cfg.DATA.IMAGE.CHANNELS, pretrain=True)

        if cfg.MODEL.SSL.CHECKPOINT_PATH is not None:
            self.model.load_state_dict(torch.load(cfg.MODEL.SSL.CHECKPOINT_PATH)['state_dict'])

        self.num_patches = (self.model.swinModel.config.image_size // self.model.swinModel.config.patch_size) ** 2
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.MODEL.SSL.TRAINING.LEARNING_RATE, weight_decay=cfg.MODEL.SSL.TRAINING.WEIGHT_DECAY)

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, image, batch_size):
        mask_positions = torch.randint(low=0, high=2, size=(batch_size, self.num_patches)).bool()
        mask_positions = mask_positions.to(self.device)
        outputs = self.model(pixel_values= image, bool_masked_pos=mask_positions)
        loss, reconstructed_pixel_values = outputs[0], outputs[1]
        return loss, reconstructed_pixel_values
    
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
    dataset = LoFiDataset(dataset_path=cfg.DATA.PATH)
    
    train_size = int(cfg.MODEL.SSL.TRAINING.TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])


    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.MODEL.SSL.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.MODEL.SSL.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=cfg.MODEL.SSL.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.MODEL.SSL.VALIDATION.NUM_WORKERS, persistent_workers=True)

    model = PretrainImageEncoder(cfg)
    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        strategy="auto",
        max_epochs=cfg.MODEL.SSL.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.MODEL.SSL.TRAINING.SWA_LRS)], 
        default_root_dir="/data/hkerner/NASA-MLCommons/lofi"
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)