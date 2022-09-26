"""Project global constants."""

#  Dataset information
ORIGINALS = "/home/mimi/Projects/own_projects/aerial_image_segmentation/data/kaggle/dataset/semantic_drone_dataset/original_images"
MASKS = "/home/mimi/Projects/own_projects/aerial_image_segmentation/data/kaggle/dataset/semantic_drone_dataset/label_images_semantic"
ORIGINAL_TYPE = ".jpg"
MASK_TYPE = ".png"
SEGMENTATION_CLASSES = 24

#  Project structure information
NAME = "test"
DATA_ROOT = "./data"
MODELS_FOLDER = "models"
TRAININGS_DATA_FOLDER = "trainings_data"
PLOTS_FOLDER = "plots"

#  U-Net parameters
RESHAPE = [192, 256]
BATCH = 16
EPOCHS = 100
TEST_TRAIN_VALIDATION = [0.8, 0.1, 0.1]
CONVOLUTION_KERNEL_SIZE = [5, 5]
MAX_POOL_SIZE = [2, 2]
FILTERS_NUMBER = [32, 64, 128, 256, 512]
LEARNING_RATE = 1e-3
EPSILON = 1e-7
