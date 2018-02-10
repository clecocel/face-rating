from keras.models import load_model
import tensorflow as tf
from data_generator import main

def test(filename):
    training_generator, training_samples, test_set, test_samples = main(
        batch_size=BATCH_SIZE, data_augmentation=False, test_split=1)
    x_test, y_test = test_set
    sess = tf.Session()
    model = load_model(filename + '.h5')
    preds = model.predict(x_test, batch_size=1)
    with open('stats.txt', 'w') as f:
        print([p[0] for p in preds], file=f)

test('train_last_layers_mse_augmentation_on/results_model10_adam_lr0.001_run1_last20')