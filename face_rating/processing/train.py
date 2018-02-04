#!/usr/bin/env python3

from data_generator import main
from resnet import *
from keras.optimizers import SGD

BATCH_SIZE = 128

training_generator, training_samples, test_set, test_samples = main(batch_size=BATCH_SIZE)

def write_results(filename, history):
    with open(filename, 'w') as f:
        print(history.history['mean_absolute_error'], file=f)
        print(history.history['val_mean_absolute_error'], file=f)

def train(model, filename=None):
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=20,
        callbacks=None,
        validation_data=test_set)
    if filename is not None:
        write_results(filename, history)
    return history

train(model9, 'results_model9.txt')
train(model3, 'results_model3.txt')

'''
results = []

results.append(train(model1))
write_results('results_model1.txt', results[-1])
results.append(train(model2))
write_results('results_model2.txt', results[-1])
results.append(train(model3))
write_results('results_model3.txt', results[-1])
results.append(train(model4))
write_results('results_model4.txt', results[-1])
results.append(train(model5))
write_results('results_model5.txt', results[-1])
results.append(train(model6))
write_results('results_model6.txt', results[-1])
results.append(train(model7))
write_results('results_model7.txt', results[-1])
results.append(train(model8))
write_results('results_model8.txt', results[-1])


with open('training_results.txt', 'w') as f:
    for result in results:
        print(result.history['mean_absolute_error'], file=f)
        print(result.history['val_mean_absolute_error'], file=f)
        print('------------------', file=f)
'''



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