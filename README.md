# U-Net for image semantic segmentation
This project contain U-Net architecture in response to closed Kaggle competition [Aerial Semantic Segmentation Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset), which is based on [this](http://dronedataset.icg.tugraz.at) dataset. This code is able to extract segments of a given input image.

## Setup
To use unet project, `pip` dependencies must be installed. It can be done via following command: 
```commandline
pip install -r requirements.txt -r dev-requirements.txt
```
It is also recommended to use `pyenv` to work with, `python 3.10` with all required dependencies in one place. Activation of environment can be done by:
```commandline
pyenv install 3.10.4
pyenv virtualenv 3.10.4 unet
pyenv activate unet
```
Last step is to build the project by calling `setup.py` file with:
```commandline
pip install -e .
```
After this operation all internal imports in `unet` module folder will see each other.

## Training
To run training, fulfill `unet/constants.py` file, with proper paths to downloaded [dataset](http://dronedataset.icg.tugraz.at). After this, and after dependencies installation, you can run training e.g. by typing in terminal:
```commandline
python unet/training.py
```

## Notes
With given default parameters `unet` is actually able to hit `~60%` of accuracy. Further work on the accuracy is ongoing.