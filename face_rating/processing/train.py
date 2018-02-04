#!/usr/bin/env python3

from data_generator import main
from resnet import model
from keras.optimizers import SGD

BATCH_SIZE = 64

training_generator, training_samples, test_set, test_samples = main(batch_size=BATCH_SIZE)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

history = model.fit_generator(
    training_generator,
    steps_per_epoch=training_samples // BATCH_SIZE,
    epochs=1,
    callbacks=None,
    validation_data=test_set)

print(history.history['mean_absolute_error'])
print(history.history['val_mean_absolute_error'])
'''
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='mean_squared_error', metrics=['mae', 'mse'])

model.fit_generator(
    training_generator,
    steps_per_epoch=training_samples // BATCH_SIZE,
    epochs=10,
    callbacks=None,
    validation_data=test_set)

model.save('resnet.h5')
'''