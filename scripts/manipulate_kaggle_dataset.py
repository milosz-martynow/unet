"""This script introduce possibility to manipulate Kaggle dataset.
Dataset link: https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset
This is helping tool, thus variables are defined here instead of constants.py

Main purpose of this script is to manipulate Kaggle structured image base, e.g.
- Extract certain classes we want
- Convert them into grayscale

For proper work of those script it is needed to place under
/data/images/kaggle_dataset/semantic_drone_dataset
all available Kaggle data (from above link):
- label_images_semantic
- RGB_color_image_masks
- original_images
- class_dict_seg.xls

This dataset contain classes with RGB maps as follows:
name	    r   g   b
unlabeled	0	0	0
paved-area	128	64	128
dirt	    130	76	0
grass	    0	102	0
gravel	    112	103	87
water	    28	42	168
rocks	    48	41	30
pool	    0	50	89
vegetation	107	142	35
roof	    70	70	70
wall	    102	102	156
window	    254	228	12
door	    254	148	12
fence	    190	153	153
fence-pole	153	153	153
person	    255	22	96
dog	        102	51	0
car	        9	143	150
bicycle	    119	11	32
tree	    51	51	0
bald-tree	190	250	190
ar-marker	112	150	146
obstacle	2	135	115
conflicting	255	0	0
"""
from typing import Dict
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

SEGMENTS: str = "data/images/kaggle_dataset/semantic_drone_dataset/class_dict_seg.xls"
RGB_MASKS: str = (
    "data/images/kaggle_dataset/semantic_drone_dataset/RGB_color_image_masks"
)
OPERATION: Dict[str, dict] = {
    "extract_single_segment": {
        "Run": True,
        "class_name": "paved-area",
        "name": "binary_masks_paved-area",
    }
}


def load_segments_dict(segments_file: Path) -> pd.DataFrame:
    """Load segments RGB values.
    :param segments_file: .xls file with segments RGB pixel values coding.
    :type segments_file: str
    :returns: Dataframe of RGB pixel values coding.
    :rtype: pd.DataFrame
    """
    segments = pd.read_table(segments_file, header=0, delimiter=", ", engine="python")
    segments.set_index("name", inplace=True)
    segments = segments[["b", "g", "r"]]
    print(segments)
    return segments


def extract_single_segment(
    segments_file: str, masks_root: str, segment_name: str, save_path: Path
):
    """Main function for extracting single segment and save it as binary image.
    :param segments_file: .xls file with segments RGB pixel values coding.
    :type segments_file: str
    :param masks_root: RGB masks images folder - coupled with segments file.
    :type masks_root: str
    :param segment_name: Name of segment to extract.
    :type segment_name: str
    :param save_path: Path where output images will be saved.
    :type save_path: Path
    """
    segments_pixels = load_segments_dict(segments_file=segments_file)
    segment = segments_pixels.loc[segment_name].to_numpy().reshape((1, 1, 3)).flatten()

    masks_paths = [p for p in sorted(masks_root.glob("*.png"))]
    for path in masks_paths:
        save_name = str(Path(save_path, path.name))
        print(save_name)
        image = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
        image = np.where(image == segment, 255, 0).astype(np.uint8)
        cv2.imwrite(save_name, image)


if __name__ == "__main__":

    segments_file = Path(SEGMENTS)
    rgb_masks_root = Path(RGB_MASKS)

    if OPERATION["extract_single_segment"]["Run"]:
        save_root = Path(RGB_MASKS).parent
        save_root = Path(save_root, OPERATION["extract_single_segment"]["name"])
        save_root.mkdir(parents=True, exist_ok=True)

        extract_single_segment(
            segments_file=segments_file,
            segment_name=OPERATION["extract_single_segment"]["class_name"],
            masks_root=rgb_masks_root,
            save_path=save_root,
        )
