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
    #  If originals and masks are converted to normalized grayscale images (pixels in range 0-1),
    #  single convolution is sufficient to extract main features.
    #  In case of 3 channels images, it is recommended to have 3 convolution layers.

    convolution = tf.keras.layers.Conv2D(
        filters=features_in_layer,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.LecunNormal,
    )(convolution)

    #  Batch normalization prevents loss being huge
    convolution = tf.keras.layers.BatchNormalization()(convolution)

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

    skip_connections = []
    for filters in features_in_layer:
        convolution = _convolution_block(
            convolution=convolution, features_in_layer=filters, kernel_size=kernel_size
        )

        skip_connections.append(convolution)

        if filters != features_in_layer[-1]:
            #  Do not use max pooling in the bottom layer
            convolution = tf.keras.layers.MaxPool2D(pool_size)(convolution)

    return convolution, skip_connections


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

        convolution = tf.keras.layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size, strides=pool_size, padding="same"
        )(convolution)
        convolution = tf.keras.layers.Concatenate()([convolution, skip])
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
    convolution_base = tf.keras.layers.Input(shape=(*reshape, 1))

    convolution, skip_connections = _encoder(
        convolution=convolution_base,
        features_in_layer=features_in_layer,
        kernel_size=kernel_size,
        pool_size=pool_size,
    )

    features_in_layer.reverse()
    skip_connections.reverse()
    features_in_layer = features_in_layer[1::]
    skip_connections = skip_connections[1::]

    convolution = _decoder(
        convolution=convolution,
        skip_connection=skip_connections,
        features_in_layer=features_in_layer,
        kernel_size=kernel_size,
        pool_size=pool_size,
    )

    #  Apply non-linearity via sigmoid at the end, normalize the output
    #  Thus, there is no need to have from_logits=True in loss function.
    #  Nevertheless, above might lead to situation where training params
    #  stuck at some level, due to small non-linearity.
    #  Applying relu, with from_logits=True in loss function might fix that.
    convolution = tf.keras.layers.Conv2D(
        segmentation_classes, kernel_size=(1, 1), padding="same", activation="relu"
    )(convolution)

    return tf.keras.models.Model(inputs=convolution_base, outputs=convolution)
