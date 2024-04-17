from yacs.config import CfgNode as CN

_C = CN()

# Data Sequence
_C.DATA = CN()
_C.DATA.GPS_GALLERY = "D:/kerner-lab/geo-clip-paper-implementation/coordinates_100K.csv"
_C.DATA.PATH = "/data/hkerner/NASA-MLCommons/lofi_dataset"

_C.DATA.IMAGE = CN()
_C.DATA.IMAGE.SIZE = 512
_C.DATA.IMAGE.CHANNELS = 6


# Model
_C.MODEL = CN()
_C.MODEL.SEED_VALUE = 43

# Self Supervised Learning Training
_C.MODEL.SSL = CN()
_C.MODEL.SSL.CHECKPOINT_PATH = "/data/hkerner/NASA-MLCommons/lofi/lightning_logs/version_15119571/checkpoints/epoch=13-step=9478.ckpt"
_C.MODEL.SSL.TRAINING = CN()
_C.MODEL.SSL.TRAINING.LEARNING_RATE = 3e-5
_C.MODEL.SSL.TRAINING.WEIGHT_DECAY = 1e-6
_C.MODEL.SSL.TRAINING.BATCH_SIZE = 64
_C.MODEL.SSL.TRAINING.MAX_EPOCHS = 200
_C.MODEL.SSL.TRAINING.NUM_WORKERS = 8
_C.MODEL.SSL.TRAINING.SWA_LRS = 1e-3
_C.MODEL.SSL.TRAINING.TRAIN_SPLIT = 0.8

# Self Supervised Learning Validation
_C.MODEL.SSL.VALIDATION = CN()
_C.MODEL.SSL.VALIDATION.BATCH_SIZE = 32
_C.MODEL.SSL.VALIDATION.NUM_WORKERS = 4


# Contrastive Learning Training
_C.MODEL.CL = CN()
_C.MODEL.CL.CHECKPOINT_PATH = "/data/hkerner/NASA-MLCommons/lofi/lightning_logs/version_15119571/checkpoints/epoch=13-step=9478.ckpt"
_C.MODEL.CL.GPS_QUEUE_SIZE = 992

# Supervised Learning Training
_C.MODEL.SL = CN()
_C.MODEL.SL.CHECKPOINT_PATH = "/data/hkerner/NASA-MLCommons/lofi/lightning_logs/version_15119571/checkpoints/epoch=13-step=9478.ckpt"
