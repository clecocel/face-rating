#!/usr/bin/env python3

from data_generator import main
from resnet import *
from keras.optimizers import SGD, Adam
from train_report_generator import generate_train_report

BATCH_SIZE = 32

training_generator, training_samples, test_set, test_samples = main(
    batch_size=BATCH_SIZE, data_augmentation=False, test_split=0.4)

def write_results(filename, history, **kwargs):
    if isinstance(history, list):
        training_error = []
        for hist in history:
            training_error += hist.history['mean_absolute_error']
        validation_error = []
        for hist in history:
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




PATH_PREFIX = 'train_first_pass_mse'
model_nbs = [9, 10, 11, 12]
opt = 'adam'


# learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
learning_rates = [0.001]

for run in range(1, 11):
    for model_nb in model_nbs:
        for lr in learning_rates:
            print("Training Model {} - learning rate {} - run {}".format(model_nb, lr, run))
            train(
                make_model(model_nb),
                filename='./{}/results_model{}_{}_lr{}_run{}'.format(PATH_PREFIX, model_nb, opt, lr, run),
                lr=lr,
                epochs=30,
                optimizer=opt)





'''
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
'''
