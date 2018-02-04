#!/usr/bin/env python3

from data_generator import main
from resnet import *
from keras.optimizers import SGD

BATCH_SIZE = 128

training_generator, training_samples, test_set, test_samples = main(batch_size=BATCH_SIZE)



def train(model):
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=50,
        callbacks=None,
        validation_data=test_set)
    return history

results = []

results.append(train(model1))
results.append(train(model2))
results.append(train(model3))
results.append(train(model4))
results.append(train(model5))
results.append(train(model6))
results.append(train(model7))
results.append(train(model8))

with open('training_results.txt', 'w') as f:
    for result in results:
        print(result.history['mean_absolute_error'], file=f)
        print(result.history['val_mean_absolute_error'], file=f)
        print('------------------', file=f)

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