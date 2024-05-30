## XAI based NST_Augmentation with Horovod

Initially installed Tensorflow Version: 2.13.0+nv23.8

## Packages to be installed
import os
import time
import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow import keras(Version: 2.13.1)
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Conv2D
import PIL

## Parameters to be set
learning_rate=0.02
beta_1=0.99
epsilon= 1e-1
epochs=10

## Dataset for Style and Content folder
https://www.onlinemedicalimages.com/index.php/en/



