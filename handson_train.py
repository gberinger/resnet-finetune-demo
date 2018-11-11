#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: train.py
# Date: 16.06.2017
# Author: Adam Brzeski - CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

"""
Performs training of a single fully-connected classifier layer on a cached set of feature vectors prepared with
build_features.py. Trained model is saved to classifier_weights.h5.
"""

import os
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD


TRAIN_DIR = os.path.expanduser("ml/data/indoor/train")
VAL_DIR = os.path.expanduser("ml/data/indoor/val")
FEATURES_FILENAME = "features-resnet152.npy"
LABELS_FILENAME = "labels-resnet152.npy"
WEIGHTS_CLASSIFIER = "classifier_weights.h5"


# Load train data
train_features = np.load(os.path.join(TRAIN_DIR, FEATURES_FILENAME))
train_labels = np.load(os.path.join(TRAIN_DIR, LABELS_FILENAME))

# Load val data
val_features = np.load(os.path.join(VAL_DIR, FEATURES_FILENAME))
val_labels = np.load(os.path.join(VAL_DIR, LABELS_FILENAME))

# TASK: Change train/val labels to one-hot versions.
train_labels = ...
val_labels = ...

# TASK: Build softmax model. It should have 1 Dense layer with 67 outputs.
classifier_model = ...
classifier_model.add(...)

# TASK: Define SGD optimizer with learning rate 0.1 and compile the model with cross-entropy loss
#       and accuracy metric.
opt = ...
classifier_model.compile(...)

# Prepare callbacks
# TASK: Add learning rate schedule callback, which reduces the learning rate by 0.9 after
#       1 epoch of not improving on validation set.
lr_decay = ...
# TASK: Add saving model checkpointer, which save the best version of model to specified file.
checkpointer = ...

# TASK: Train your model. Don't forget to add validation and callbacks!
classifier_model.fit(...)
