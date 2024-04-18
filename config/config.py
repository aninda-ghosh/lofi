from yacs.config import CfgNode as CN

_C = CN()

# Data Sequence
_C.DATA = CN()
_C.DATA.GPS_GALLERY = "/data/hkerner/NASA-MLCommons/lofi/dataset/coordinates_100K.csv"
_C.DATA.TRAIN_PATH = "/data/hkerner/NASA-MLCommons/lofi_dataset/train.csv"
_C.DATA.VALIDATION_PATH = "/data/hkerner/NASA-MLCommons/lofi_dataset/validation.csv"
_C.DATA.TEST_PATH = "/data/hkerner/NASA-MLCommons/lofi_dataset/test.csv"

_C.DATA.IMAGE = CN()
_C.DATA.IMAGE.SIZE = 512
_C.DATA.IMAGE.CHANNELS = 6


# Model
_C.MODEL = CN()
_C.MODEL.SEED_VALUE = 43

# Self Supervised Learning Training
_C.MODEL.SSL = CN()
_C.MODEL.SSL.CHECKPOINT_PATH = "/data/hkerner/NASA-MLCommons/lofi/logs/ssl/lightning_logs/version_15124617/checkpoints/epoch=3-step=5412.ckpt"
_C.MODEL.SSL.TRAINING = CN()
_C.MODEL.SSL.TRAINING.LEARNING_RATE = 3e-5
_C.MODEL.SSL.TRAINING.WEIGHT_DECAY = 1e-6
_C.MODEL.SSL.TRAINING.BATCH_SIZE = 64
_C.MODEL.SSL.TRAINING.MAX_EPOCHS = 60
_C.MODEL.SSL.TRAINING.NUM_WORKERS = 8
_C.MODEL.SSL.TRAINING.SWA_LRS = 1e-3

# Self Supervised Learning Validation
_C.MODEL.SSL.VALIDATION = CN()
_C.MODEL.SSL.VALIDATION.BATCH_SIZE = 32
_C.MODEL.SSL.VALIDATION.NUM_WORKERS = 4


# Contrastive Learning Training
_C.MODEL.CL = CN()
_C.MODEL.CL.CHECKPOINT_PATH = None
_C.MODEL.CL.GPS_QUEUE_SIZE = 800
_C.MODEL.CL.TRAINING = CN()
_C.MODEL.CL.TRAINING.LEARNING_RATE = 3e-5
_C.MODEL.CL.TRAINING.WEIGHT_DECAY = 1e-6
_C.MODEL.CL.TRAINING.BATCH_SIZE = 32
_C.MODEL.CL.TRAINING.MAX_EPOCHS = 10
_C.MODEL.CL.TRAINING.NUM_WORKERS = 8
_C.MODEL.CL.TRAINING.SWA_LRS = 1e-3

# Self Supervised Learning Validation
_C.MODEL.CL.VALIDATION = CN()
_C.MODEL.CL.VALIDATION.BATCH_SIZE = 16
_C.MODEL.CL.VALIDATION.NUM_WORKERS = 4


# Supervised Learning Training
_C.MODEL.SL = CN()
_C.MODEL.SL.CHECKPOINT_PATH = "/data/hkerner/NASA-MLCommons/lofi/lightning_logs/version_15119571/checkpoints/epoch=13-step=9478.ckpt"
