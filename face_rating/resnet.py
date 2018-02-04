#!/usr/bin/env python3

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense, Flatten
from keras.models import Model
import numpy as np

base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output

# We flatten then output 1 value (regression - maybe we should use classification)
x = Flatten()(x)
prediction = Dense(1)(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=prediction)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
