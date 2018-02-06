#!/usr/bin/env python3

from data_generator import main
from resnet import *
from keras.optimizers import SGD, Adam

BATCH_SIZE = 32

training_generator, training_samples, test_set, test_samples = main(batch_size=BATCH_SIZE, data_augmentation=False, test_split=0.4)

def write_results(filename, history):
    with open(filename, 'w') as f:
        if isinstance(history, list):
            for hist in history:
                print(hist.history['mean_absolute_error'], file=f)
            for hist in history:
                print(hist.history['val_mean_absolute_error'], file=f)
        else:
            write_results(filename, [history])

def train(model, filename=None):
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae', 'mse'])

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=30,
        callbacks=None,
        validation_data=test_set)
    if filename is not None:
        write_results(filename, history)
    return history


def train_2(model, filename=None, layers_second_pass=10):
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.95, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae', 'mse'])

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=15,
        callbacks=None,
        validation_data=test_set)
    for layer in model.layers[-layers_second_pass:]:
        layer.trainable = True

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae', 'mse'])

    history2 = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=20,
        callbacks=None,
        validation_data=test_set)

    if filename is not None:
        write_results(filename, [history, history2])
    return history

i = 7
for run in range(5, 11):
    print("Training Model {}".format(i))
    train_2(make_model(i), 'results_model{}_run{}.txt'.format(i, run), 10)
