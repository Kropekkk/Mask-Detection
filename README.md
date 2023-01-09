# Mask-Detection

Mask Detection app

Dataset: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

![Example](https://github.com/Kropekkk/Mask-Detection/blob/main/media/example2.png)

## Dependencies
* Pytorch
* OpenCV
* Django, FastAPI (2 different branches)
* Faster R-CNN ResNet 50

## Setup

1. Create virtual environment ```python -m venv enviro```
2. Activate the virtual environment```.\enviro\Scripts\activate```
3. Install dependencies ```pip install -r requirements.txt```

## Usage

1. Train the model (Remember to clone dataset and and set the parameters in config.py)

```
 python train.py
```

2. Run the app (If Django)

```
 python manage.py runserver
```