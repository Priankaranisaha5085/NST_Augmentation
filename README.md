## XAI based NST_Augmentation with MultiGpu

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
learning_rate=set the learning rate between 0 to 1,
beta_1=set the beta value in between 0 to 1,
epsilon= set the epsilon value in between 1e-1 to 1e-2,
Number of Epochs= Set the epochs

## Dataset for Style and Content folder
https://www.onlinemedicalimages.com/index.php/en/


## Classification of Augmented data
After augmentation for classification execute "Resnet classification code.m".


