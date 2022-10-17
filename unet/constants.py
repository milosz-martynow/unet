"""Project global constants."""
from typing import List

#  Dataset information
ORIGINALS: str = "data/images/kaggle_dataset/semantic_drone_dataset/original_images"
MASKS: str = "data/images/kaggle_dataset/semantic_drone_dataset/binary_masks_paved-area"
SEGMENTATION_CLASSES: int = 2

#  Project structure information
NAME: str = "test_epochs-500_binary-image_batch-8"
DATA_ROOT: str = "./data"
MODELS_FOLDER: str = "models"
TRAININGS_DATA_FOLDER: str = "trainings_data"
PLOTS_FOLDER: str = "plots"

#  U-Net parameters
RESHAPE: List[int] = [192, 256]
BATCH: int = 8
EPOCHS: int = 500
TEST_TRAIN_VALIDATION: List[float] = [0.8, 0.1, 0.1]
CONVOLUTION_KERNEL_SIZE: List[int] = [3, 3]
MAX_POOL_SIZE: List[int] = [2, 2]
FILTERS_NUMBER: List[int] = [64, 128, 256, 512, 1028]
LEARNING_RATE: float = 1e-03
EPSILON: float = 1e-07
