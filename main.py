from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import tensorflow as tf

NAME = "test100"
DATA_ROOT = "./data"
MODELS_FOLDER = "models"
TRAININGS_DATA_FOLDER = "trainings_data"
PLOTS_FOLDER = "plots"
ORIGINALS = "/home/mimi/Projects/own_projects/aerial_image_segmentation/data/kaggle/dataset/semantic_drone_dataset/original_images"
MASKS = "/home/mimi/Projects/own_projects/aerial_image_segmentation/data/kaggle/dataset/semantic_drone_dataset/label_images_semantic"
ORIGINAL_TYPE = ".jpg"
MASK_TYPE = ".png"
SEGMENTATION_CLASSES = 24
RESHAPE = [192, 256]
BATCH = 16
EPOCHS = 100
TEST_TRAIN_VALIDATION = [0.8, 0.1, 0.1]
CONVOLUTION_KERNEL_SIZE = [5, 5]
MAX_POOL_SIZE = [2, 2]
FILTERS_NUMBER = [32, 64, 128, 256, 512]
LEARNING_RATE = 1e-3
EPSILON = 1e-7


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
    originals_root: Path, masks_root: Path, originals_type: str, masks_type: str
) -> Tuple[List[str], List[str]]:
    """Function that scans given folders in search of original
    images full paths and masks full paths.
    :param originals_root: Root path of original images.
    :type originals_root: Path
    :param masks_root: Root path of masks images.
    :type masks_root: Path
    :param originals_type: Type of original images to search - just extension e.g. .jpg.
    :type originals_type: str
    :param masks_type: Type of masks images to search - just extension e.g. .png.
    :type masks_type: str
    :returns: Sorted lists of full paths of original images and masks with specific extensions.
    :rtype: Tuple[List[str], List[str]]
    """

    originals_paths = [
        str(p) for p in sorted(originals_root.glob(f"*{originals_type}"))
    ]
    masks_paths = [str(p) for p in sorted(masks_root.glob(f"*{masks_type}"))]

    return originals_paths, masks_paths


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
    original_path: tf.Tensor, mask_path: tf.Tensor, originals_type: str, masks_type: str
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Function that standardize input original and mask image into TensoFlow object.
    :param original_path: Original image path presented as Tensor object. Coupled with mask_path.
    :type original_path: tf.Tensor
    :param mask_path: Mask image path presented as Tensor object. Coupled with original_path.
    :type mask_path: tf.Tensor
    :param originals_type: Type of original images to search - just extension e.g. .jpg.
    :type originals_type: str
    :param masks_type: Type of masks images to search - just extension e.g. .png.
    :type masks_type: str
    :returns: Original and mask tensor objects of standardized form.
    :rtype: tf.data.Dataset
    """

    def _decode_image(input_image: tf.Tensor, image_type: str) -> tf.Tensor:
        """Function that handle multiple type decoding routines."""

        if image_type == ".png":
            return tf.image.decode_png(input_image, channels=3)

        elif image_type == ".jpg":
            return tf.image.decode_jpeg(input_image, channels=3)

    def _original_images_standardization(
        original_path: tf.Tensor, originals_type: str
    ) -> tf.Tensor:
        """Function standardizing original image."""

        original = tf.io.read_file(original_path)
        original = _decode_image(input_image=original, image_type=originals_type)
        original = tf.image.convert_image_dtype(original, tf.float32)

        return original

    def _masks_images_standardization(
        mask_path: tf.Tensor, masks_type: str
    ) -> tf.Tensor:
        """Function standardizing mask image."""

        mask = tf.io.read_file(mask_path)
        mask = _decode_image(input_image=mask, image_type=masks_type)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)

        return mask

    original = _original_images_standardization(original_path, originals_type)
    mask = _masks_images_standardization(mask_path, masks_type)

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
        return tf.image.resize(input_image, reshape)

    original = _preprocess(original, reshape)
    mask = _preprocess(mask, reshape)

    return original, mask


def compile_dataset(
    originals_paths: List[str],
    masks_paths: List[str],
    originals_type: str,
    masks_type: str,
    reshape: List[int],
    batch_size: int,
) -> tf.data.Dataset:
    """Function that compiles all data set processing steps.
    :param original_path: Original image path presented as Tensor object. Coupled with mask_path.
    :type original_path: List[str]
    :param mask_path: Mask image path presented as Tensor object. Coupled with original_path.
    :type mask_path: List[str]
    :param originals_type: Type of original images to search - just extension e.g. .jpg.
    :type originals_type: str
    :param masks_type: Type of masks images to search - just extension e.g. .png.
    :type masks_type: str
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

    dataset = dataset.map(
        lambda original, mask: standardize_input_type(
            original, mask, originals_type=originals_type, masks_type=masks_type
        )
    )

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

    convolution = tf.keras.layers.Dropout(0.5)(convolution)

    return convolution


def _encoder(
    convolution: Union[tf.keras.layers.Input, tf.keras.layers.Conv2D],
    skip_connection: List,
    features_in_layer: List[int],
    kernel_size: List[int],
    pool_size: List[int],
) -> Tuple[tf.keras.layers.Conv2D, List[tf.keras.layers.Conv2D]]:
    """Function encoding the input image into set of features. Downsampling of image features.
    :param convolution: Input image to be convolved. Might be image of convolution.
    :type convolution: Union[tf.keras.layers.Input, tf.keras.layers.Conv2D]
    :param skip_connection: last convolution of _convolution_block of each filter iteration.
    :type skip_connection: List
    :param features_in_layer: Number of dimensions at the output of convolution.
    :type features_in_layer: int
    :param kernel_size: Size of convolution window.
    :type kernel_size: List[int]
    :param pool_size: Size of convolution window.
    :type pool_size: List[int]
    :returns: Convolution object and skip connection layers list.
    :rtype: Tuple[tf.keras.layers.Conv2D, List[tf.keras.layers.Conv2D]]
    """

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
        convolution = tf.keras.layers.Conv2DTranspose(
            filters, kernel_size=kernel_size, strides=pool_size, padding="same"
        )(convolution)
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
    skip_connection = []

    convolution, skip_connection = _encoder(
        convolution=convolution_base,
        skip_connection=skip_connection,
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


if __name__ == "__main__":

    data_root, model_path, training_data_path, plots_path = prepare_project_structure(
        data_root=DATA_ROOT,
        models_path=MODELS_FOLDER,
        training_data_path=TRAININGS_DATA_FOLDER,
        plots_path=PLOTS_FOLDER,
    )

    originals, masks = dataset_paths(originals_root=ORIGINALS, masks_root=MASKS)

    originals_paths, masks_paths = input_images_and_masks_paths(
        originals_root=originals,
        masks_root=masks,
        originals_type=ORIGINAL_TYPE,
        masks_type=MASK_TYPE,
    )

    dataset = compile_dataset(
        originals_paths=originals_paths,
        masks_paths=masks_paths,
        originals_type=ORIGINAL_TYPE,
        masks_type=MASK_TYPE,
        reshape=RESHAPE,
        batch_size=BATCH,
    )

    plot_original_and_mask(
        dataset=dataset, dataset_number=1, prefix=NAME, save_path=plots_path
    )

    train_dataset, test_dataset, validation_dataset = split_dataset(
        dataset=dataset,
        train_size=TEST_TRAIN_VALIDATION[0],
        test_size=TEST_TRAIN_VALIDATION[1],
        validation_size=TEST_TRAIN_VALIDATION[2],
    )

    model = build_model(
        features_in_layer=FILTERS_NUMBER,
        reshape=RESHAPE,
        kernel_size=CONVOLUTION_KERNEL_SIZE,
        pool_size=MAX_POOL_SIZE,
        segmentation_classes=SEGMENTATION_CLASSES,
    )

    model.summary()

    """initial_learning_rate * decay_rate ^ (step / decay_steps)"""
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE, decay_steps=10000, decay_rate=0.9
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=EPSILON),
        metrics=["accuracy"],
    )

    model_name, train_data = prepare_output_files_names(
        prefix=NAME, model_path=model_path, training_data_path=training_data_path
    )

    train_steps, validation_steps = number_of_steps(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=BATCH,
    )

    model_history = model.fit(
        x=train_dataset.repeat(),
        validation_data=validation_dataset.repeat(),
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(model_name),
            tf.keras.callbacks.CSVLogger(train_data),
        ],
        steps_per_epoch=train_steps,
        validation_steps=validation_steps,
    )

    plot_metrics(model_history=model_history, prefix=NAME, save_path=plots_path)

    plot_predicted_mask(
        dataset=test_dataset,
        model=model,
        dataset_number=1,
        prefix=NAME,
        save_path=plots_path,
    )
