from yacs.config import CfgNode as CN

_C = CN()

# Data Sequence
_C.DATA = CN()
# _C.DATA.GPS_GALLERY = "D:/kerner-lab/geo-clip-paper-implementation/coordinates_100K.csv"
_C.DATA.TRAIN_DATASET_PATH = "/data/hkerner/NASA-MLCommons/lofi_dataset"
# _C.DATA.EVAL_DATASET_PATH = "D:/kerner-lab/datasets/yfcc4k/"
#
_C.IMAGE =CN()
_C.IMAGE.SIZE = 512
_C.IMAGE.CHANNELS = 6


# Model
_C.MODEL = CN()
# _C.MODEL.CHECKPOINT_PATH = "D:/kerner-lab/geo-clip-paper-implementation/lightning_logs/version_2/checkpoints/epoch=5-step=17172.ckpt"
_C.MODEL.GPS_QUEUE_SIZE = 992
_C.MODEL.SEED_VALUE = 43

# Training
_C.TRAINING = CN()
_C.TRAINING.LEARNING_RATE = 3e-5
_C.TRAINING.WEIGHT_DECAY = 1e-6
_C.TRAINING.BATCH_SIZE = 64
_C.TRAINING.MAX_EPOCHS = 200
_C.TRAINING.NUM_WORKERS = 8
_C.TRAINING.SWA_LRS = 1e-3
_C.TRAINING.TRAIN_SPLIT = 0.8

# Validation
_C.VALIDATION = CN()
_C.VALIDATION.BATCH_SIZE = 32
_C.VALIDATION.NUM_WORKERS = 4