"""Test utilities scripts."""

import pytest
from pathlib import Path
from typing import List
import shutil
from unet.utilities import (
    prepare_project_structure,
    prepare_output_files_names,
    dataset_paths,
    input_images_and_masks_paths,
    number_of_steps,
)


@pytest.mark.parametrize(
    "data_root, models_path, training_data_path, plots_path",
    [
        pytest.param(
            "tests/test_data",
            "tests/models",
            "tests/trainings_data",
            "tests/plots",
            id="Project data folders test.",
        ),
    ],
)
def test_prepare_project_structure(
    data_root: str, models_path: str, training_data_path: str, plots_path: str
):
    """Test whether folders are created correctly."""
    models_path, training_data_path, plots_path = prepare_project_structure(
        data_root, models_path, training_data_path, plots_path
    )

    assert models_path.exists()
    assert training_data_path.exists()
    assert plots_path.exists()

    #  Clean up
    shutil.rmtree(data_root)


@pytest.mark.parametrize(
    "prefix, model_path, training_data_path",
    [
        pytest.param(
            "test",
            "model",
            "training_data",
            id="Project files names test.",
        ),
    ],
)
def test_prepare_output_files_names(
    prefix: str, model_path: str, training_data_path: str
):
    """Test whether data names are created correctly."""

    model = Path(model_path, f"{prefix}_model.h5")
    training_data = Path(training_data_path, f"{prefix}_training.csv")

    assert model, training_data == prepare_output_files_names(
        prefix, model_path, training_data_path
    )


@pytest.mark.parametrize(
    "originals_root, masks_root",
    [
        pytest.param(
            "/home/mimi/originals",
            "/home/mimi/masks",
            id="Linux paths test.",
        ),
        pytest.param(
            "c:\\projects\\originals",
            "c:\\projects\\masks",
            id="Windows paths test.",
        ),
    ],
)
def test_dataset_paths(originals_root: str, masks_root: str):
    """Test whether created paths are system independent."""
    original_path = Path(originals_root)
    mask_path = Path(masks_root)
    assert original_path, mask_path == dataset_paths(originals_root, masks_root)


@pytest.mark.parametrize(
    "root, originals, masks",
    [
        pytest.param(
            "tests",
            [""],
            [""],
            id="Folder scan test - elements: 0.",
        ),
        pytest.param(
            "tests",
            ["original_0.jpg"],
            ["mask_0.png"],
            id="Folder scan test - elements: 1.",
        ),
        pytest.param(
            "tests",
            ["original_0.jpg", "original_1.jpg"],
            ["mask_0.png", "mask_1.png"],
            id="Folder scan test - elements: 2.",
        ),
    ],
)
def test_input_images_and_masks_paths(
    root: str, originals: List[str], masks: List[str]
):
    """Test whether function is able to list and sort folder content."""

    originals_root = Path(root, "originals")
    masks_root = Path(root, "masks")
    originals_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    for original, mask in zip(originals, masks):

        Path(originals_root, f"{original}").touch()
        Path(masks_root, f"{mask}").touch()

        assert originals, masks == input_images_and_masks_paths(
            originals_root, masks_root
        )

    #  Clean up
    shutil.rmtree(originals_root)
    shutil.rmtree(masks_root)


@pytest.mark.parametrize(
    "dataset_size, batch_size, result",
    [
        pytest.param(
            0,
            0,
            None,
            id="Batch size test: 0, 0 -> None",
        ),
        pytest.param(
            0,
            1,
            0,
            id="Batch size test: 0, 1 -> 0",
        ),
        pytest.param(
            1,
            0,
            None,
            id="Batch size test: 1, 0 -> None",
        ),
        pytest.param(
            1,
            1,
            1,
            id="Batch size test: 1, 1 -> 1",
        ),
        pytest.param(
            0,
            2,
            0,
            id="Batch size test: 0, 2 -> 0",
        ),
        pytest.param(
            2,
            0,
            None,
            id="Batch size test: 2, 0 -> None",
        ),
        pytest.param(
            2,
            2,
            1,
            id="Batch size test: 2, 2 -> 1",
        ),
    ],
)
def test_number_of_steps(dataset_size: int, batch_size: int, result: int):
    """Test whether length of batch is correct."""
    assert result == number_of_steps(dataset_size, batch_size)
