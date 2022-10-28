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

    return tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(originals_paths, name="Originals"),
            tf.constant(masks_paths, name="Masks"),
        )
    )


def image_process(path: tf.Tensor, decode_method: str, reshape: List[int]) -> tf.Tensor:
    """Read, decode, standardize and reshape given image
    :param path: Original image path presented as Tensor object. Coupled with mask_path.
    :type path: tf.Tensor
    :param reshape: Width and height of new image.
    :type reshape: List[int]
    :param decode_method: type of image to decode: PNG or JPEG
    :type decode_method: str
    :returns: Preprocessed image - reading, decoding, standardization, reshape.
    :rtype: tf.Tensor
    """
    image = tf.io.read_file(path)

    if decode_method == "PNG":
        image = tf.image.decode_jpeg(image, channels=1)
    elif decode_method == "JPEG":
        image = tf.image.decode_jpeg(image, channels=1)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, reshape, method="nearest")

    return image


@tf.function
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

    def _preprocess_original_and_mask(
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

        original = image_process(path=original, decode_method="JPEG", reshape=reshape)
        mask = image_process(path=mask, decode_method="PNG", reshape=reshape)

        return original, mask

    dataset = tensorflow_dataset(
        originals_paths=originals_paths, masks_paths=masks_paths
    )

    dataset = dataset.map(
        lambda original, mask: _preprocess_original_and_mask(
            original, mask, reshape=reshape
        )
    )

    return dataset.batch(batch_size)


def split_dataset(
    dataset: tf.data.Dataset,
    split_sizes: List[float],
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Function that splits dataset containing originals with masks images,
    into train, test, validation datasets. At the begging input dataset is shuffled.
    Test, train and validation floats, have to sum up to 1.0.
    :param dataset: Tensorflow dataset object, connecting original and masks images with them self.
    :type dataset: tf.data.Dataset
    :param split_sizes: Train, test and validation dataset sizes, from 0.0 to 1.0.
    :type split_sizes: float
    :returns: Train, test and validation dataset.
    :rtype: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
    """

    dataset_size = len(dataset)

    dataset = dataset.shuffle(buffer_size=int(dataset_size / 10))

    n_train = int(dataset_size * split_sizes[0])
    n_test = int(dataset_size * split_sizes[2])
    n_validation = int(dataset_size * split_sizes[1])

    train_dataset = dataset.take(n_train)
    test_dataset = dataset.skip(n_train + n_validation).take(n_test)
    validation_dataset = dataset.skip(n_train).take(n_validation)

    return train_dataset, test_dataset, validation_dataset
