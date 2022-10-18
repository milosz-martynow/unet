"""Project global constants."""
from typing import List

#  Dataset information
ORIGINALS: str = "data/images/kaggle_dataset/semantic_drone_dataset/original_images"
MASKS: str = "data/images/kaggle_dataset/semantic_drone_dataset/binary_masks_paved-area"
SEGMENTATION_CLASSES: int = 1

#  Project structure information
NAME: str = "test_epochs-10_binary-images_relu-output-logits-true"
DATA_ROOT: str = "./data"
MODELS_FOLDER: str = "models"
TRAININGS_DATA_FOLDER: str = "trainings_data"
PLOTS_FOLDER: str = "plots"

#  U-Net parameters
RESHAPE: List[int] = [256, 288]
BATCH: int = 4
EPOCHS: int = 10
TRAIN_TEST_VALIDATION: List[float] = [0.8, 0.1, 0.1]
CONVOLUTION_KERNEL_SIZE: List[int] = [3, 3]
MAX_POOL_SIZE: List[int] = [2, 2]
FILTERS_NUMBER: List[int] = [64, 128, 256, 512, 1028]
LEARNING_RATE: float = 1e-02
