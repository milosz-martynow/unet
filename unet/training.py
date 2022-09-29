"""Run training."""

import tensorflow as tf

from unet.constants import (
    BATCH,
    CONVOLUTION_KERNEL_SIZE,
    DATA_ROOT,
    EPOCHS,
    EPSILON,
    FILTERS_NUMBER,
    LEARNING_RATE,
    MASKS,
    MAX_POOL_SIZE,
    MODELS_FOLDER,
    NAME,
    ORIGINALS,
    PLOTS_FOLDER,
    RESHAPE,
    SEGMENTATION_CLASSES,
    TEST_TRAIN_VALIDATION,
    TRAININGS_DATA_FOLDER,
)
from unet.dataset import compile_dataset, split_dataset
from unet.model import build_model
from unet.plots import plot_metrics, plot_predicted_mask
from unet.utilities import (
    dataset_paths,
    input_images_and_masks_paths,
    number_of_steps,
    prepare_output_files_names,
    prepare_project_structure,
)

if __name__ == "__main__":

    model_path, training_data_path, plots_path = prepare_project_structure(
        data_root=DATA_ROOT,
        models_path=MODELS_FOLDER,
        training_data_path=TRAININGS_DATA_FOLDER,
        plots_path=PLOTS_FOLDER,
    )

    originals, masks = dataset_paths(originals_root=ORIGINALS, masks_root=MASKS)

    originals_paths, masks_paths = input_images_and_masks_paths(
        originals_root=originals, masks_root=masks
    )

    dataset = compile_dataset(
        originals_paths=originals_paths,
        masks_paths=masks_paths,
        reshape=RESHAPE,
        batch_size=BATCH,
    )

    train_dataset, test_dataset, validation_dataset = split_dataset(
        dataset=dataset,
        split_sizes=TEST_TRAIN_VALIDATION,
    )

    model = build_model(
        features_in_layer=FILTERS_NUMBER,
        reshape=RESHAPE,
        kernel_size=CONVOLUTION_KERNEL_SIZE,
        pool_size=MAX_POOL_SIZE,
        segmentation_classes=SEGMENTATION_CLASSES,
    )

    model.summary()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE, epsilon=EPSILON
        ),
        metrics=["accuracy"],
    )

    model_name, train_data = prepare_output_files_names(
        prefix=NAME, model_path=model_path, training_data_path=training_data_path
    )

    train_steps = number_of_steps(
        dataset_size=len(train_dataset),
        batch_size=BATCH,
    )
    validation_steps = number_of_steps(
        dataset_size=len(validation_dataset),
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
