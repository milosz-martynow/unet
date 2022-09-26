"""Prepare tensorflow dataset object to feed training process."""

from typing import List, Tuple

import tensorflow as tf


def tensorflow_dataset(
    originals_paths: List[str], masks_paths: List[str]
) -> tf.data.Dataset:
    """Function that creates single TensorFlow dataset object with concatenated paths.
    :param originals_paths: List with original images full paths. Connected in order with masks_paths.
    :type originals_paths: List[str]
    :param masks_paths: List with masks images full paths. Connected in order with images_paths.
    :type masks_paths: List[str]
    :returns: Tensorflow dataset object, connecting original and masks images with them self.
    :rtype: tf.data.Dataset
    """

    return tf.data.Dataset.from_tensor_slices((originals_paths, masks_paths))


def standardize_input_type(
    original_path: tf.Tensor, mask_path: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Function that standardize input original and mask image into TensoFlow object.
    :param original_path: Original image path presented as Tensor object. Coupled with mask_path.
    :type original_path: tf.Tensor
    :param mask_path: Mask image path presented as Tensor object. Coupled with original_path.
    :type mask_path: tf.Tensor
    :returns: Original and mask tensor objects of standardized form.
    :rtype: tf.data.Dataset
    """

    original = tf.io.read_file(original_path)
    original = tf.image.decode_jpeg(original, channels=3)
    original = tf.image.convert_image_dtype(original, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1, dtype=tf.dtypes.uint8)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)

    return original, mask


def preprocess(
    original: tf.Tensor, mask: tf.Tensor, reshape: List[int]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Function that preprocess input images, to create robust images set.
    :param original: Original image, coupled with mask.
    :type original: tf.Tensor
    :param mask: Mask image of coupled original image.
    :type mask: tf.Tensor
    :param reshape: Width and height of new image.
    :type reshape: List[int]
    :returns: Preprocessed original and mask image.
    :rtype: Tuple[tf.Tensor, tf.Tensor]
    """

    def _preprocess(input_image: tf.Tensor, reshape: List[int]):
        """Encapsulated preprocessing function to keep the same procedures for original and mask."""
        return tf.image.resize(input_image, reshape, method="nearest")

    original = _preprocess(original, reshape)
    mask = _preprocess(mask, reshape)

    return original, mask


def compile_dataset(
    originals_paths: List[str],
    masks_paths: List[str],
    reshape: List[int],
    batch_size: int,
) -> tf.data.Dataset:
    """Function that compiles all data set processing steps.
    :param originals_paths: Original image path presented as Tensor object. Coupled with mask_path.
    :type originals_paths: List[str]
    :param masks_paths: Mask image path presented as Tensor object. Coupled with original_path.
    :type masks_paths: List[str]
    :param reshape: Width and height of new image.
    :type reshape: List[int]
    :param batch_size: Number of images in single batch.
    :type batch_size: int
    :returns: Dataset with all original and masks images coupled in pairs.
    :rtype: tf.data.Dataset
    """

    dataset = tensorflow_dataset(
        originals_paths=originals_paths, masks_paths=masks_paths
    )

    dataset = dataset.map(lambda original, mask: standardize_input_type(original, mask))

    dataset = dataset.map(
        lambda original, mask: preprocess(original, mask, reshape=reshape)
    )

    return dataset.batch(batch_size)


def split_dataset(
    dataset: tf.data.Dataset,
    train_size: float,
    validation_size: float,
    test_size: float,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Function that splits dataset containing originals with masks images,
    into train, test, validation datasets. At the begging input dataset is shuffled.
    Test, train and validation floats, have to sum up to 1.0.
    :param dataset: Tensorflow dataset object, connecting original and masks images with them self.
    :type dataset: tf.data.Dataset
    :param train_size: Train dataset size, from 0.0 to 1.0.
    :type train_size: float
    :param validation_size: Validation dataset size, from 0.0 to 1.0.
    :type validation_size: float
    :param test_size: Test dataset size, from 0.0 to 1.0.
    :type test_size: float
    :returns: Train, test and validation dataset.
    :rtype: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
    """

    dataset_size = len(dataset)

    dataset = dataset.shuffle(buffer_size=int(dataset_size / 10))

    n_train = int(dataset_size * train_size)
    n_validation = int(dataset_size * validation_size)
    n_test = int(dataset_size * test_size)

    train_dataset = dataset.take(n_train)
    validation_dataset = dataset.skip(n_train).take(n_validation)
    test_dataset = dataset.skip(n_train + n_validation).take(n_test)

    return train_dataset, test_dataset, validation_dataset
