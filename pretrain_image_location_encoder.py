import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import StochasticWeightAveraging
import math
import pandas as pd
import numpy as np

from dataset.location_pretrain_dataset import LocationPretrainDataset
from model.image_encoder import ImageEncoder
from model.location_encoder import LocationEncoder

from config import cfg


class Contrastive_Loss(nn.Module):
    def __init__(self):
        super(Contrastive_Loss, self).__init__()

    def forward(self, cosine_sim_matrix):
        logits = cosine_sim_matrix

        exp_logits = torch.exp(logits)        
            
        diag_logits = torch.diag(exp_logits)

        #get the sum of the exponential of the logits
        exp_logits_sum = exp_logits.sum(1)

        #compute the loss
        loss = -torch.log(diag_logits / exp_logits_sum)

        #compute the mean loss
        loss = loss.mean()

        return loss


class PretrainLocationEncoder(pl.LightningModule):
    def __init__(self, cfg):
        super(PretrainLocationEncoder, self).__init__()

        self.cfg = cfg
        self.image_encoder = ImageEncoder(image_size=cfg.DATA.IMAGE.SIZE, num_channels=cfg.DATA.IMAGE.CHANNELS, freeze=True)


        # TODO: Since we have frozen the image encoder we can load the weights from the supervised learning model
        if cfg.MODEL.SSL.CHECKPOINT_PATH:
            state_dict = torch.load(cfg.MODEL.SSL.CHECKPOINT_PATH)["state_dict"]
            # only load the image encoder weights and not the decoder weights
            state_dict = {k.replace("image_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder")}
            
            self.image_encoder.load_state_dict(state_dict=state_dict)



        self.location_encoder = LocationEncoder(embedding_size=256)

        self.num_patches = (self.image_encoder.swinModel.config.image_size // self.image_encoder.swinModel.config.patch_size) ** 2
        
        self.criterion = Contrastive_Loss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=cfg.MODEL.CL.TRAINING.LEARNING_RATE, weight_decay=cfg.MODEL.CL.TRAINING.WEIGHT_DECAY)


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gps_gallery = torch.tensor(pd.read_csv(cfg.DATA.GPS_GALLERY)[['latitude', 'longitude']].values, dtype=torch.float32) 
        self.queue_size = cfg.MODEL.CL.GPS_QUEUE_SIZE
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, image, gps):        
        image_embedding = self.image_encoder(pixel_values=image)
        sequence_output = image_embedding[0]
        sequence_output = sequence_output.transpose(1, 2)
        image_features = sequence_output.sum(dim=1)
        
        gps_embedding = self.location_encoder(gps)
        gps_features = F.normalize(gps_embedding, dim=1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_features @ gps_features.t())
        return logits_per_image
    
    def training_step(self, batch):
        image, gps = batch

        gps_queue = self.get_gps_queue()

        gps_all = torch.cat([gps, gps_queue], dim=0)
        self.dequeue_and_enqueue(gps)

        output = self(image, gps_all)

        loss = self.criterion(output)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch):
        image, gps = batch

        gps = torch.tensor(gps, dtype=torch.float32)
        gps_queue = self.get_gps_queue()

        gps_all = torch.cat([gps, gps_queue], dim=0)
        
        output = self(image, gps_all)
        
        loss = self.criterion(output)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


if __name__ == "__main__":
    pl.seed_everything(cfg.MODEL.SEED_VALUE, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    

    torch.set_float32_matmul_precision('medium')

    # Initialize the dataset and dataloaders
    train_dataset = LocationPretrainDataset(dataset_path=cfg.DATA.TRAIN_PATH)
    val_dataset = LocationPretrainDataset(dataset_path=cfg.DATA.VALIDATION_PATH)
    
    train_dataloader = DataLoader(train_dataset, pin_memory=True, batch_size=cfg.MODEL.CL.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.MODEL.CL.TRAINING.NUM_WORKERS, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, batch_size=cfg.MODEL.CL.VALIDATION.BATCH_SIZE, shuffle=False, num_workers=cfg.MODEL.CL.VALIDATION.NUM_WORKERS, persistent_workers=True)

    model = PretrainLocationEncoder(cfg)

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        strategy="auto",
        max_epochs=cfg.MODEL.CL.TRAINING.MAX_EPOCHS,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=[StochasticWeightAveraging(swa_lrs=cfg.MODEL.CL.TRAINING.SWA_LRS)], 
        default_root_dir="/data/hkerner/NASA-MLCommons/lofi/logs/cl"
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=cfg.MODEL.CL.CHECKPOINT_PATH)