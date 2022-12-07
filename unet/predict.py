from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from constants import RESHAPE
from dataset import image_process

ORIGINAL_PATH: str = "data/images/dataset/originals/DJI_0001.jpg"
MODEL_PATH: str = "data/models/epochs-500_init-LecunNormal_extra-images_model.h5"
PLOT: Optional[str] = "./aaa.png"
MASK_PATH: Optional[
    str
] = None  # "data/images/kaggle_dataset/semantic_drone_dataset/binary_masks_paved-area/109.png"


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


def postprocessing(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ).astype(np.uint8)
    _, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    image = cv2.dilate(image, kernel, iterations=2)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image


def single_prediction(
    image_path: str,
    model_path: str,
    plot: Optional[str],
    reshape: List[int] = RESHAPE,
    mask_path: Optional[str] = None,
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
    #  mask_predicted = postprocessing(image=mask_predicted)
    original = image_process(path=image_path, decode_method="PNG", reshape=reshape)

    if plot is not None:
        if mask_path is not None:
            mask = image_process(path=mask_path, decode_method="PNG", reshape=reshape)
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            ax3.imshow(mask)
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        ax1.imshow(original)
        ax2.imshow(mask_predicted)
        fig.savefig(plot)


if __name__ == "__main__":
    single_prediction(
        image_path=ORIGINAL_PATH, mask_path=MASK_PATH, model_path=MODEL_PATH, plot=PLOT
    )
