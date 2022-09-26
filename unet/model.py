"""U-Net model architecture."""
from typing import List, Tuple, Union

import tensorflow as tf


def _convolution_block(
    convolution: Union[tf.keras.layers.Input, tf.keras.layers.Conv2D],
    features_in_layer: int,
    kernel_size: List[int],
) -> tf.keras.Input:
    """Neural network convolution single block. the same for encoding and decoding.
    :param convolution: Input image to be convolved. Might be image of convolution.
    :type convolution: Union[tf.keras.layers.Input, tf.keras.layers.Conv2D]
    :param features_in_layer: Number of dimensions at the output of convolution.
    :type features_in_layer: int
    :param kernel_size: Size of convolution window.
    :type kernel_size: List[int]
    :returns: Convolution object.
    :rtype: tf.keras.Input
    """

    convolution = tf.keras.layers.Conv2D(
        filters=features_in_layer,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal,
    )(convolution)

    convolution = tf.keras.layers.Conv2D(
        filters=features_in_layer,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal,
    )(convolution)

    convolution = tf.keras.layers.Conv2D(
        filters=features_in_layer,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.HeNormal,
    )(convolution)

    convolution = tf.keras.layers.Dropout(0.1)(convolution)

    return convolution


def _encoder(
    convolution: Union[tf.keras.layers.Input, tf.keras.layers.Conv2D],
    features_in_layer: List[int],
    kernel_size: List[int],
    pool_size: List[int],
) -> Tuple[tf.keras.layers.Conv2D, List[tf.keras.layers.Conv2D]]:
    """Function encoding the input image into set of features. Downsampling of image features.
    :param convolution: Input image to be convolved. Might be image of convolution.
    :type convolution: Union[tf.keras.layers.Input, tf.keras.layers.Conv2D]
    :param features_in_layer: Number of dimensions at the output of convolution.
    :type features_in_layer: int
    :param kernel_size: Size of convolution window.
    :type kernel_size: List[int]
    :param pool_size: Size of convolution window.
    :type pool_size: List[int]
    :returns: Convolution object and skip connection layers list.
    :rtype: Tuple[tf.keras.layers.Conv2D, List[tf.keras.layers.Conv2D]]
    """

    skip_connection = []
    for filters in features_in_layer:
        convolution = _convolution_block(
            convolution=convolution, features_in_layer=filters, kernel_size=kernel_size
        )
        skip_connection.append(convolution)
        convolution = tf.keras.layers.MaxPool2D(pool_size)(convolution)

    return convolution, skip_connection


def _decoder(
    convolution: tf.keras.layers.Conv2D,
    skip_connection: List[tf.keras.Input],
    features_in_layer: List[int],
    kernel_size: List[int],
    pool_size: List[int],
) -> tf.keras.layers.Conv2D:
    """Function decoding the convolution image into map of features. Upsampling of image features.
    :param convolution: Input image to be convolved. Might be image of convolution.
    :type convolution: tf.keras.layers.Conv2D
    :param skip_connection: last convolution of _convolution_block of each filter iteration.
    :type skip_connection: List[tf.keras.layers.Conv2D]
    :param features_in_layer: Number of dimensions at the output of convolution.
    :type features_in_layer: int
    :param kernel_size: Size of convolution window.
    :type kernel_size: List[int]
    :param pool_size: Size of convolution window.
    :type pool_size: List[int]
    :returns: Convolution object.
    :rtype: tf.keras.layers.Conv2D
    """
    for skip, filters in zip(skip_connection, features_in_layer):
        convolution = tf.keras.layers.UpSampling2D(size=pool_size)(convolution)
        convolution = tf.keras.layers.Concatenate(axis=3)([convolution, skip])
        convolution = _convolution_block(
            convolution=convolution, features_in_layer=filters, kernel_size=kernel_size
        )

    return convolution


def build_model(
    features_in_layer: List[int],
    reshape: List[int],
    kernel_size: List[int],
    pool_size: List[int],
    segmentation_classes: int,
) -> tf.keras.models:
    """Function that creates U-net model.
    :param features_in_layer: Number of dimensions at the output of convolution.
    :type features_in_layer: int
    :param reshape: Width and height of new image.
    :type reshape: List[int]
    :param kernel_size: Size of convolution window.
    :type kernel_size: List[int]
    :param pool_size: Size of convolution window.
    :type pool_size: List[int]
    :param segmentation_classes: Number of features we expect to segment in the image.
    :type segmentation_classes: int
    :returns: U-net model architecture
    :rtype: tf.keras.models
    """
    convolution_base = tf.keras.layers.Input(shape=(*reshape, 3))

    convolution, skip_connection = _encoder(
        convolution=convolution_base,
        features_in_layer=features_in_layer,
        kernel_size=kernel_size,
        pool_size=pool_size,
    )

    features_in_layer.reverse()
    skip_connection.reverse()

    convolution = _decoder(
        convolution=convolution,
        skip_connection=skip_connection,
        features_in_layer=features_in_layer,
        kernel_size=kernel_size,
        pool_size=pool_size,
    )

    convolution = tf.keras.layers.Conv2D(
        segmentation_classes, kernel_size=(1, 1), padding="same", activation="relu"
    )(convolution)

    return tf.keras.models.Model(inputs=convolution_base, outputs=convolution)
