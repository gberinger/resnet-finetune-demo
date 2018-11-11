#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: test.py
# Date: 16.06.2017
# Author: Adam Brzeski - CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

"""
Tests the trained classifier on the cached features and labels of the test set.
"""

import os
import json
from collections import defaultdict
import numpy as np
from keras.layers import Dense
from keras.models import Sequential


WEIGHTS_CLASSIFIER = "classifier_weights.h5"
TEST_DIR = os.path.expanduser("ml/data/indoor/test")
FEATURES_FILENAME = "features-resnet152.npy"
LABELS_FILENAME = "labels-resnet152.npy"
IDS_TO_NAMES = json.load(open("ids_to_names.json"))

# Load test data
test_features = np.load(os.path.join(TEST_DIR, FEATURES_FILENAME))
test_labels = np.load(os.path.join(TEST_DIR, LABELS_FILENAME))

# TASK: Load top layer classifier model. To do this, you have to specify the same model as in train.py
#       and the load the weights from checkpoint.
classifier_model = ...
classifier_model.add(...)
...

# Classify the test set, count correct answers
all_count = defaultdict(int)
correct_count = defaultdict(int)
for code, correct_label in zip(test_features, test_labels):

    code = np.expand_dims(code, axis=0)
    # TASK: Use classifier to predict a probability distribution over classes.
    #       As a result, choose the most probable label.
    prediction = ...
    result = ...

    # Record which class occured and wheter the prediction was successful
    all_count[correct_label] += 1
    if correct_label == result:
        correct_count[correct_label] += 1

# Calculate accuracies
print("Accuracy per class:")
for classid in all_count.keys():
    print("\t{}: {:.5f}".format(IDS_TO_NAMES[str(classid)], correct_count[classid] / all_count[classid]))

# TASK: Calculate average per class accuracy and overall accuracy
print("Average per class acc: {:.5f}".format(...))
print("Overall acc: {:.5f}".format(...))
