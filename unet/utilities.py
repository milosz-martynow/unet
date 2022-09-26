"""Utilities scripts"""

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf


def prepare_project_structure(
    data_root: str, models_path: str, training_data_path: str, plots_path: str
) -> Tuple[Path, Path, Path]:
    """Function that prepare project structure. Creates folder if its needed,
    and returns paths to them in Path objects
    :param data_root: Root folder of overall data storage.
    :type data_root: str
    :param models_path: Sub-folder name for models archivision from root.
    :type models_path: str
    :param training_data_path: Sub-folder name for training data archivision from root.
    :type training_data_path: str
    :param plots_path: Sub-folder name for plots archivision from root.
    :type plots_path: str
    :returns: Paths to root of data storage, models and training epochs information.
    :rtype: List[Path, Path, Path, Path]
    """

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    models_path = Path(data_root, models_path)
    models_path.mkdir(parents=True, exist_ok=True)
    training_data_path = Path(data_root, training_data_path)
    training_data_path.mkdir(parents=True, exist_ok=True)
    plots_path = Path(data_root, plots_path)
    plots_path.mkdir(parents=True, exist_ok=True)

    return data_root, models_path, training_data_path, plots_path


def prepare_output_files_names(
    prefix: str, model_path: Path, training_data_path: Path
) -> Tuple[Path, Path]:
    """Function preparing data files Path, with root paths.
    :param prefix: Descriptive prefix standing in the file name.
    :type prefix: str
    :param models_path: Sub-folder name for models archivision.
    :type models_path: Path
    :param training_data_path: Sub-folder path for training data archivision.
    :type training_data_path: Path
    :returns: Paths to root of data storage, models and training epochs information.
    :rtype: Tuple[Path, Path]
    """
    model = Path(model_path, f"{prefix}_model.h5")
    training_data = Path(training_data_path, f"{prefix}_training.csv")

    return model, training_data


def dataset_paths(originals_root: Path, masks_root: Path) -> Tuple[Path, Path]:
    """Function that create system independent Path objects pointing to the
    folders with originals images and corresponding to them masks.
    :param originals_root: Root path of original images.
    :type originals_root: Path
    :param masks_root: Root path of masks images.
    :type masks_root: Path
    :returns: PAth object with root paths to originals and masks images.
    :rtype: Tuple[Path, Path]
    """
    return Path(originals_root), Path(masks_root)


def input_images_and_masks_paths(
    originals_root: Path, masks_root: Path
) -> Tuple[List[str], List[str]]:
    """Function that scans given folders in search of original
    images full paths and masks full paths.
    :param originals_root: Root path of original images.
    :type originals_root: Path
    :param masks_root: Root path of masks images.
    :type masks_root: Path
    :returns: Sorted lists of full paths of original images and masks with specific extensions.
    :rtype: Tuple[List[str], List[str]]
    """

    originals_paths = [str(p) for p in sorted(originals_root.glob("*.jpg"))]
    masks_paths = [str(p) for p in sorted(masks_root.glob("*.png"))]

    tf.data.Dataset.from_tensor_slices((originals_paths, masks_paths))

    return originals_paths, masks_paths


def number_of_steps(
    train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, batch_size: int
) -> Tuple[int, int]:
    """Calculate number of steps to take per each epoch.
    :param train_dataset: Train images dataset.
    :type train_dataset: tf.data.Dataset
    :param validation_dataset: Validation images dataset
    :type validation_dataset: tf.data.Dataset
    :param batch_size: Number of images in single batch.
    :type batch_size: int
    :returns: Number of steps to take in each epoch by training and validation process.
    :rtype: Tuple[int, int]
    """

    train_steps = len(train_dataset) // batch_size
    validation_steps = len(validation_dataset) // batch_size

    if len(train_dataset) % batch_size != 0:
        train_steps += 1
    if len(validation_dataset) % batch_size != 0:
        validation_steps += 1

    return train_steps, validation_steps
