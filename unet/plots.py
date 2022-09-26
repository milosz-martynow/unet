"""Quick plots scripts."""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


def plot_original_and_mask(
    dataset: tf.data.Dataset, dataset_number: int, prefix: str, save_path: Path
):
    """Helper function that generate plot object with original and corresponding to it mask images.
    It saves the image in the project root instead of pop up showing.
    :param dataset: Tensorflow dataset object, connecting original and masks images with them self.
    :type dataset: tf.data.Dataset
    :param dataset_number: Number of dataset to plot - starts from 1.
    :type dataset_number: int
    :param prefix: Descriptive prefix standing in the file name.
    :type prefix: str
    :param save_path: full path for save the image.
    :type save_path: Path
    """

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    for original, mask in dataset.take(dataset_number):
        ax1.imshow(original.numpy()[0])
        ax2.imshow(mask.numpy()[0])
    fig.savefig(
        Path(
            f"{save_path}",
            f"{prefix}_image_number_{dataset_number}_original_and_mask.png",
        )
    )


def plot_predicted_mask(
    dataset: tf.data.Dataset,
    model: tf.keras.models,
    dataset_number: int,
    prefix: str,
    save_path: Path,
):
    """Helper function that generate plot object with original and corresponding to it mask images.
    It saves the image in the project root instead of pop up showing.
    :param dataset: Tensorflow dataset object, connecting original and masks images with them self.
    :type dataset: tf.data.Dataset
    :param dataset_number: Number of dataset to plot - starts from 1.
    :type dataset_number: int
    :param prefix: Descriptive prefix standing in the file name.
    :type prefix: str
    :param save_path: full path for save the image.
    :type save_path: Path
    :param: U-net model architecture (trained model).
    :type: tf.keras.models
    """

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    for original, mask in dataset.take(dataset_number):

        predicted_mask = model.predict(original)
        predicted_mask = tf.argmax(predicted_mask, axis=-1)
        predicted_mask = predicted_mask[..., tf.newaxis]

        ax1.imshow(original.numpy()[dataset_number])
        ax2.imshow(mask.numpy()[dataset_number])
        ax3.imshow(predicted_mask[dataset_number])

        fig.savefig(
            Path(
                f"{save_path}",
                f"{prefix}_image_number_{dataset_number}_predicted_mask.png",
            )
        )


def plot_metrics(model_history: tf.keras.callbacks, prefix: str, save_path: Path):
    """Function that prints metrics in the function of epochs.
    :param model_history: History of the model callbacks.
    :type model_history: tf.keras.callbacks
    :param prefix: Descriptive prefix standing in the file name.
    :type prefix: str
    :param save_path: full path for save the image.
    :type save_path: Path
    """

    fig, ax1 = plt.subplots(ncols=1, nrows=1)
    ax1.plot(model_history.history["accuracy"])
    fig.savefig(Path(f"{save_path}", f"{prefix}_metrics.png"))
