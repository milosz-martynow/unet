# U-Net for image semantic segmentation
This project contain U-Net architecture in response to closed Kaggle competition [Aerial Semantic Segmentation Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset), which is based on [this](http://dronedataset.icg.tugraz.at) dataset. This code is able to extract segments of a given input image.

## Setup
To use unet project, `pip` dependencies must be installed. It can be done via following command: 
```commandline
pip install -r requirements.txt -r dev-requirements.txt
```
Last step is to build the project by calling `setup.py` file with:
```commandline
pip install -e .
```
After this operation all internal imports in `unet` module folder will see each other.

It is also recommended to use `pyenv` to work with, `python 3.10` with all required dependencies in one place. Activation of environment can be done by:
```commandline
pyenv install 3.10.4
pyenv virtualenv 3.10.4 unet
pyenv activate unet
```
For PyCharm users - after this go to `Settings | Project | Python Interpreter | Add Interpreter | Add Local Interpreter | Virtual Environment | New | and hoose location of interpreter`. Now IDE will see all pip installed packages - including in Python Console.
## Training
To run training, fulfill `unet/constants.py` file, with proper paths to downloaded [dataset](http://dronedataset.icg.tugraz.at). After this, and after dependencies installation, you can run training e.g. by typing in terminal:
```commandline
python unet/training.py
```

## Makefile
Keep code clean and standardized by using `isort` (sorting imports), `black` (automatic code formatting) and `pylint` (checks code standards). To do everything at once you can use two options of `Makefile`:
```commandline
make lint 
```
for longer execution including `pylint`, or
```commandline
make format
```
To get fast format of your code with usage only `isort` and `black`.

`Makefile` gives an option to perform installation procedure via:
```commandline
make install
```

Toy can also run automatically tests with:
```commandline
make test
```

## Notes
 - With given default parameters, `unet` is actually able to hit `~70%` of accuracy at `100` epochs. Further work on the accuracy is ongoing.
 - With 3x3 convolution kernel, with Lecun Normal initializer, single class segmentation reach 98% (paved area).
 - Base U-Net mode developed on publication [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
 - Data is tested on:
   - Kaggle competition on [Aerial Semantic Segmentation Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)