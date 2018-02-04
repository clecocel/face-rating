#!/usr/bin/env python3

from data_generator import main
from resnet import model

training_gen, training_samples, test_set, test_samples = main(batch_size=64)

model.fit_generator(
    training_generator,
    steps_per_epoch=int(training_samples / batch_size),
    epochs=1,
    callbacks=None,
    validation_data=test_set)
