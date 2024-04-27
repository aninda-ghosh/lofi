import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging
import math
from lightning.pytorch.utilities import grad_norm

from dataset.semantic_segmentation_dataset import LoFiSemanticDataset

from model.image_encoder import ImageEncoder
from model.semantic_mask_decoder import MaskDecoder
from model.location_encoder import LocationEncoder

from solver.losses import DiceLoss

from config import cfg

class LoFi(pl.LightningModule):
    def __init__(self, cfg):
        super(LoFi, self).__init__()

        self.cfg = cfg
        self.image_encoder = ImageEncoder(image_size=cfg.DATA.IMAGE.SIZE, num_channels=cfg.DATA.IMAGE.CHANNELS, freeze=True)

        self.location_encoder = LocationEncoder(embedding_size=256, freeze=True)

        # TODO: Since we have frozen the image encoder we can load the weights from the self supervised learning model
        if cfg.MODEL.PIL.CHECKPOINT_PATH:
            state_dict = torch.load(cfg.MODEL.PIL.CHECKPOINT_PATH)["state_dict"]
            # only load the image encoder weights and not the decoder weights
            image_encoder_state_dict = {k.replace("image_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder")}
            location_encoder_state_dict = {k.replace("location_encoder.", ""): v for k, v in state_dict.items() if k.startswith("location_encoder")}
            
            self.image_encoder.load_state_dict(state_dict=image_encoder_state_dict)
            self.location_encoder.load_state_dict(state_dict=location_encoder_state_dict)

        self.mask_decoder = MaskDecoder(input_channels=769, out_channels=1)

        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.MODEL.SS.TRAINING.LEARNING_RATE, weight_decay=cfg.MODEL.SS.TRAINING.WEIGHT_DECAY)

    def configure_optimizers(self):
        return self.optimizer
    
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def forward(self, image, gps, batch_size):
        image_embedding = self.image_encoder(pixel_values=image, bool_masked_pos=None)
        sequence_output = image_embedding[0]
        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output.transpose(1, 2)
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width)

        # Get the GPS features from the location encoder
        gps_embedding = self.location_encoder(gps)
        gps_embedding = F.normalize(gps_embedding, dim=1)
        gps_embedding = gps_embedding.reshape(batch_size, 1, height, width)

        # Concatenate the image and GPS embeddings
        joint_embedding = torch.cat([sequence_output, gps_embedding], dim=1)

        mask_logits = self.mask_decoder(joint_embedding)

        return mask_logits
    
    def training_step(self, batch, batch_idx):
        image, gps, target_masks = batch
        
        mask_logits = self(image, gps, image.shape[0])

        _bce_loss = self.bce_loss(mask_logits, target_masks)
        mask_logits = torch.where(mask_logits > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        _dice_loss = self.dice_loss(mask_logits, target_masks)

        loss = _bce_loss + _dice_loss

        loss = loss.mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, gps, target_masks = batch
        
        mask_logits = self(image, gps, image.shape[0])

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
    
    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.MODEL.SS.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.MODEL.SS.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=cfg.MODEL.SS.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.MODEL.SS.VALIDATION.NUM_WORKERS, persistent_workers=True)

    print("Train dataset length: ", len(train_dataset))
    print("Validation dataset length: ", len(val_dataset))

    print("Train dataloader length: ", len(train_dataloader))
    print("Validation dataloader length: ", len(val_dataloader))

    model = LoFi(cfg)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        strategy="auto",
        max_epochs=cfg.MODEL.SS.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.MODEL.SS.TRAINING.SWA_LRS)], 
        default_root_dir="/data/hkerner/NASA-MLCommons/lofi/logs/semantic_segmentation"
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg.MODEL.SS.CHECKPOINT_PATH)