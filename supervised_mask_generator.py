import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging

from dataset.image_pretrain_dataset import LoFiDataset
from model.image_encoder import ImageEncoder
from model.location_encoder import LocationEncoder
from model.mask_decoder import MaskDecoder

from config import cfg

class LoFi(pl.LightningModule):
    def __init__(self, cfg):
        super(LoFi, self).__init__()

        self.cfg = cfg
        self.image_encoder = ImageEncoder(image_size=cfg.DATA.IMAGE.SIZE, num_channels=cfg.DATA.IMAGE.CHANNELS, pretrain=False)
        self.location_encoder = LocationEncoder()
        self.mask_decoder = MaskDecoder()

        self.num_patches = (self.model.swinModel.config.image_size // self.model.swinModel.config.patch_size) ** 2
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.MODEL.SSL.TRAINING.LEARNING_RATE, weight_decay=cfg.MODEL.SSL.TRAINING.WEIGHT_DECAY)

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, image, location, batch_size):
        image_embeddings = self.image_encoder(pixel_values= image, bool_masked_pos=None)
        location_embeddings = self.location_encoder(location)

        #stack image and location embeddings
        image_embeddings = torch.cat((image_embeddings, location_embeddings), dim=1)

        mask_logits = self.mask_decoder(image_embeddings, location_embeddings, batch_size)
        return mask_logits
    
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

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg.MODEL.SSL.CHECKPOINT_PATH)