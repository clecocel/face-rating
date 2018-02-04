#!/usr/bin/env python2.7

from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten

import numpy as np

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')

x = base_model.output

# let's add a fully-connected layer
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
x = Dropout(0.4)(x)
#x = Flatten()(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.05),
                activity_regularizer=regularizers.l1(0.05))(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=prediction)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
