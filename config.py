import torch

DATA_ROOT = "data"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train/images"
TRAIN_MASK_DIR = f"{DATA_ROOT}/train/masks"
VAL_IMG_DIR = f"{DATA_ROOT}/val/images"
VAL_MASK_DIR = f"{DATA_ROOT}/val/masks"
TEST_IMG_DIR = f"{DATA_ROOT}/test/images"
TEST_MASK_DIR = f"{DATA_ROOT}/test/masks"

CHECKPOINT_DIR = "models/checkpoints"
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
NUM_CLASSES = 1          # binary building mask; set to 4 for multiclass urban version
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"