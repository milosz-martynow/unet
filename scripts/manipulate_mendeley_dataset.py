"""This script introduce possibility to manipulate dataset downloaded from:
https://data.mendeley.com/datasets/t576ydh9v8/3
Passos, Bianka T.; Cassaniga, Mateus J.; Fernandes, Anita M. R. ; Medeiros, Kátya B. ;
Comunello, Eros (2020), “Cracks and Potholes in Road Images”, Mendeley Data, V2, doi: 10.17632/t576ydh9v8.2

This dataset contains 2235 images of roads, taken from some height at some angle.
Thus, its car images dataset rather than drone.

Each folder contain following image class:
_RAW
_LANE
_POTHOLE
_CRACK

This script allows to copy all images of single class of dataset,
to some folder (e.g. originals and masks).
"""
from pathlib import Path
from typing import List
import shutil
import cv2
import numpy as np

DATASET: str = "data/images/all_datasets/mendeley"
MASKS: str = "data/images/dataset_for_binary_segmentation/masks"
ORIGINALS: str = "data/images/dataset_for_binary_segmentation/originals"
CLASS_MASKS: List[str] = ["_POTHOLE", "_CRACK"]
CLASS_ORIGINAL: str = "_RAW"


def files_of_interests(folder_to_scan: Path, class_to_search: str) -> List[Path]:
    """Finds all files in folder_to_scan directory tree,
    which contain given class_to_search sentence.
    :param folder_to_scan: Folder, which directory tree will be scanned.
    :type folder_to_scan: Path
    :param class_to_search: Sentence which will be searched in the files names.
    :type class_to_search: str
    :returns: List of files Paths that contain class_to_search in folder_to_scan directory tree
    :rtype: List[Path]
    """
    return [file for file in sorted(folder_to_scan.rglob(f"*{class_to_search}*"))]


if __name__ == "__main__":

    dataset = Path(DATASET)
    originals = Path(ORIGINALS)
    masks = Path(MASKS)

    originals.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    #  Move originals to other folder and rename it to match with masks
    original_images = files_of_interests(
        folder_to_scan=dataset, class_to_search=CLASS_ORIGINAL
    )
    for file in original_images:
        shutil.copy(
            file,
            Path.joinpath(
                originals,
                file.name.replace(CLASS_ORIGINAL, ""),
            ),
        )

    #  Merge masks to one image under the same name as its original
    masks_paths = [
        files_of_interests(folder_to_scan=dataset, class_to_search=mask_class)
        for mask_class in CLASS_MASKS
    ]
    for i in range(len(original_images)):
        new_mask = sum(
            [
                cv2.imread(str(masks_paths[j][i]), flags=cv2.IMREAD_GRAYSCALE)
                for j in range(len(CLASS_MASKS))
            ]
        )
        new_mask = np.where(new_mask >= 1, 255, 0).astype(np.uint8)
        cv2.imwrite(
            str(
                Path.joinpath(
                    masks,
                    masks_paths[0][i].name.replace(CLASS_MASKS[0], ""),
                )
            ),
            new_mask,
        )
