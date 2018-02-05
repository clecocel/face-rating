#!/usr/bin/env python2.7

from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten

import numpy as np

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')
base_model2 = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False
for layer in base_model2.layers:
    layer.trainable = False



#############################################################
x = base_model.output
# let's add a fully-connected layer
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
#x = Flatten()(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.02),
                activity_regularizer=regularizers.l1(0.02))(x)
# this is the model we will train
model1 = Model(inputs=base_model.input, outputs=prediction)
#############################################################


#############################################################
x = base_model.output
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.02),
                activity_regularizer=regularizers.l1(0.02))(x)
model2 = Model(inputs=base_model.input, outputs=prediction)
#############################################################


#############################################################
x = base_model.output
#x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.02),
                activity_regularizer=regularizers.l1(0.02))(x)
# this is the model we will train
model3 = Model(inputs=base_model.input, outputs=prediction)
#############################################################


#############################################################
x = base_model.output
#x = Flatten()(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.02),
                activity_regularizer=regularizers.l1(0.02))(x)
# this is the model we will train
model4= Model(inputs=base_model.input, outputs=prediction)
#############################################################




#############################################################
x = base_model2.output
x = Flatten()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
#x = Flatten()(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.02),
                activity_regularizer=regularizers.l1(0.02))(x)
# this is the model we will train
model5 = Model(inputs=base_model2.input, outputs=prediction)
#############################################################


#############################################################
x = base_model2.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
model6 = Model(inputs=base_model2.input, outputs=prediction)
#############################################################


#############################################################
x = base_model2.output
x = Flatten()(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.02),
                activity_regularizer=regularizers.l1(0.02))(x)
# this is the model we will train
model7 = Model(inputs=base_model2.input, outputs=prediction)
#############################################################


#############################################################
x = base_model2.output
x = Flatten()(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.02),
                activity_regularizer=regularizers.l1(0.02))(x)
# this is the model we will train
model8= Model(inputs=base_model2.input, outputs=prediction)
#############################################################

#############################################################
x = base_model2.output
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
x = Dropout(0.4)(x)
#x = Flatten()(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.05),
                activity_regularizer=regularizers.l1(0.05))(x)
model9 = Model(inputs=base_model2.input, outputs=prediction)
#############################################################

#############################################################
x = base_model2.output
x = Flatten()(x)
prediction = Dense(1, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(x)
# this is the model we will train
model10 = Model(inputs=base_model2.input, outputs=prediction)
#############################################################


