#!/usr/bin/env python3

from data_generator import main
from resnet import model

BATCH_SIZE = 64

training_generator, training_samples, test_set, test_samples = main(batch_size=BATCH_SIZE)

model.fit_generator(
    training_generator,
    steps_per_epoch=training_samples // BATCH_SIZE,
    epochs=10,
    callbacks=None,
    validation_data=test_set)
