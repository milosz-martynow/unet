from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from constants import RESHAPE
from dataset import image_process

ORIGINAL_PATH: str = "/home/mimi/Projects/own_projects/aerial_image_segmentation/kaggle/unet/data/images/kaggle_dataset/semantic_drone_dataset/original_images/109.jpg"
MASK_PATH: str = "/home/mimi/Projects/own_projects/aerial_image_segmentation/kaggle/unet/data/images/kaggle_dataset/semantic_drone_dataset/binary_masks_paved-area/109.png"
MODEL_PATH: str = "/home/mimi/Projects/own_projects/aerial_image_segmentation/kaggle/unet/data/models/test_model.h5"
PLOT: bool = True


def mask_prediction(
    image_path: str,
    model_path: str,
    decode_method: str = "JPEG",
    reshape: List[int] = RESHAPE,
) -> np.ndarray:
    """Make prediction on original image with tensorflow prediction method
    :param image_path: Path to prediction image.
    :type image_path: str
    :param model_path: Path to trained model.
    :type model_path: str
    :param decode_method: type of image to decode: PNG or JPEG
    :type decode_method: str
    :param reshape: Width and height of new image.
    :type reshape: List[int]
    :returns: predicted mask of input image via given model.
    :rtype: np.ndarray
    """
    original = np.expand_dims(
        image_process(path=image_path, decode_method=decode_method, reshape=reshape),
        axis=0,
    )
    model = tf.keras.models.load_model(model_path)

    return np.squeeze(model.predict(x=original))


def prediction(
    image_path: str,
    mask_path: str,
    model_path: str,
    plot: Optional[str],
    reshape: List[int] = RESHAPE,
):
    """Make full prediction sequence.
    :param image_path: Path to prediction image.
    :type image_path: str
    :param mask_path: Path to original mask image.
    :type mask_path: str
    :param model_path: Path to trained model.
    :type model_path: str
    :param plot: Path with name to save result.
    :type plot: Optional[str]
    :param reshape: Width and height of new image.
    :type reshape: List[int]
    :returns: predicted mask of input image via given model.
    :rtype: np.ndarray
    """
    mask_predicted = mask_prediction(image_path=image_path, model_path=model_path)
    original = image_process(path=image_path, decode_method="PNG", reshape=reshape)
    mask = image_process(path=mask_path, decode_method="PNG", reshape=reshape)

    if plot is not None:

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        ax1.imshow(original)
        ax2.imshow(mask)
        ax3.imshow(mask_predicted)
        fig.savefig(plot)


if __name__ == "__main__":
    prediction(
        image_path=ORIGINAL_PATH, mask_path=MASK_PATH, model_path=MODEL_PATH, plot=PLOT
    )
