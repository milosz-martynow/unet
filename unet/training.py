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
    MASK_TYPE,
    MASKS,
    MAX_POOL_SIZE,
    MODELS_FOLDER,
    NAME,
    ORIGINAL_TYPE,
    ORIGINALS,
    PLOTS_FOLDER,
    RESHAPE,
    SEGMENTATION_CLASSES,
    TEST_TRAIN_VALIDATION,
    TRAININGS_DATA_FOLDER,
)
from unet.dataset import compile_dataset, split_dataset
from unet.model import build_model
from unet.plots import plot_metrics, plot_original_and_mask, plot_predicted_mask
from unet.utilities import (
    dataset_paths,
    input_images_and_masks_paths,
    number_of_steps,
    prepare_output_files_names,
    prepare_project_structure,
)

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

    #  initial_learning_rate * decay_rate ^ (step / decay_steps)
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
