#!/usr/bin/env python3

from data_generator import main
from resnet import *
from keras.optimizers import SGD, Adam
from keras.models import load_model
from train_report_generator import generate_train_report

BATCH_SIZE = 32

training_generator, training_samples, test_set, test_samples = main(
    batch_size=BATCH_SIZE, data_augmentation=True, test_split=0.4)

def write_results(filename, history, **kwargs):
    if isinstance(history, list):
        training_error = []
        for hist in history:
            training_error += hist.history['mean_absolute_error']
        validation_error = []
        for hist in history:
            if 'val_mean_absolute_error' in hist.history:
                validation_error += hist.history['val_mean_absolute_error']
        report = generate_train_report(training_error, validation_error, filename, **kwargs)
        with open('{}.html'.format(filename), 'w') as f:
            f.write(report)
    else:
        write_results(filename, [history], **kwargs)

def train(model, filename=None, optimizer='adam', lr=0.001, decay=0., epochs=20, loss='mean_squared_error'):
    if optimizer == 'adam':
        opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
    if optimizer == 'sgd':
        opt = SGD(lr=lr, decay=decay)

    model.compile(optimizer=opt, loss=loss, metrics=['mae', 'mse'])

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=epochs,
        callbacks=None,
        validation_data=test_set)

    if filename is not None:
        kwargs = {
            'num_epoch': epochs,
            'fine_tuned_layers': 0,
            'optimizer': optimizer,
            'learning_rate': lr,
            'decay': decay,
        }
        write_results(filename, history, **kwargs)
    return history

def train_2(model, filename=None, optimizer='adam', lr=0.001, decay=0., epochs=20, loss='mean_squared_error', train_last_layers=1):
    if optimizer == 'adam':
        opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
    if optimizer == 'sgd':
        opt = SGD(lr=lr, decay=decay)

    model.compile(optimizer=opt, loss=loss, metrics=['mae', 'mse'])

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=epochs,
        callbacks=None,
        validation_data=test_set)

    for layer in model.layers[-train_last_layers:]:
        layer.trainable = True

    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)
    model.compile(optimizer=opt, loss=loss, metrics=['mae', 'mse'])

    history_2 = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=10,
        callbacks=None,
        validation_data=test_set)

    if filename is not None:
        kwargs = {
            'num_epoch': epochs,
            'fine_tuned_layers': 0,
            'optimizer': optimizer,
            'learning_rate': lr,
            'decay': decay,
        }
        write_results(filename, [history, history_2], **kwargs)
        model.save(filename + '.h5')
    return history


def fine_tune(filename):
    model = load_model(filename + '.h5')

    training_generator, training_samples, test_set, test_samples = main(
        batch_size=BATCH_SIZE, data_augmentation=True, test_split=0)

    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.05, amsgrad=False)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae', 'mse'])

    history = model.fit_generator(
        training_generator,
        steps_per_epoch=training_samples // BATCH_SIZE,
        epochs=20,
        callbacks=None)

    kwargs = {
        'num_epoch': 20,
        'optimizer': 'adam',
        'learning_rate': '0.0005',
        'decay': '0.05',
    }
    write_results(filename + 'fine_tuned', history, **kwargs)
    model.save(filename + 'fine_tuned.h5')
    pass


PATH_PREFIX = 'train_last_layers_mse_augmentation_on'
model_nbs = [10]
opt = 'adam'


# learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
learning_rates = [0.001]
last_layers = [10, 20]
fine_tuning = True

fine_tune_filename = PATH_PREFIX + '/results_model10_adam_lr0.001_run1_last20'

if not fine_tuning:
    for run in range(1, 11):
        for model_nb in model_nbs:
            for lr in learning_rates:
                for last_layer in last_layers:
                    print("Training Model {} - learning rate {} - run {} - last {} layers".format(
                        model_nb, lr, run, last_layer))
                    train_2(
                        make_model(model_nb),
                        filename='./{}/results_model{}_{}_lr{}_run{}_last{}'.format(
                            PATH_PREFIX, model_nb, opt, lr, run, last_layer),
                        lr=lr,
                        epochs=30,
                        optimizer=opt,
                        train_last_layers=last_layer)
else:
    fine_tune(fine_tune_filename)
