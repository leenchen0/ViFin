import os
from os.path import join as fullpath
import argparse

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

from model import (
    create_model,
    load_model_with_weights
)
from utils import (
    load_preprocessed_data,
    evaluate,
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, default='model.h5')
parser.add_argument('--data_folder', type=str, default='../data/processed/')
parser.add_argument('--train_file', type=str, default='training.mat')
parser.add_argument('--test_file', type=str, default='test.mat')
parser.add_argument('--num_key', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=220)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--training', type=bool, default=True)
args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



def train(model, data, labels, input_length, label_length, test_data, test_labels, test_input_length, test_label_length, callbacks):

    inputs = {
        'input': data,
        'labels': labels,
        'input_length': input_length,
        'label_length': label_length,
    }
    outputs = {'ctc': np.zeros([data.shape[0]])}

    validate_inputs = {
        'input': test_data,
        'labels': test_labels,
        'input_length': test_input_length,
        'label_length': test_label_length,
    }
    validate_outputs = {'ctc': np.zeros([test_data.shape[0]])}

    # training
    model.fit(
        inputs, outputs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(validate_inputs, validate_outputs),
        callbacks=callbacks
    )
    return model

def main():

    data_folder = args.data_folder

    training_filename = args.train_file
    test_filename = args.test_file
    model_name = fullpath('saved_model', args.model)

    should_training = args.training

    test_data, test_labels, test_input_length, test_label_length, num_key = load_preprocessed_data(fullpath(data_folder, test_filename))

    model = create_model(test_data[0].shape[1], num_key)

    # Training
    if should_training:
        data, labels, input_length, label_length, num_key = load_preprocessed_data(fullpath(data_folder, training_filename))
        checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_weights_only=True, save_best_only=False, mode='min', period=1)
        try:
            train(model, data, labels, input_length, label_length,
                test_data, test_labels, test_input_length, test_label_length,
                [checkpoint])
        except KeyboardInterrupt:
            # do nothing here
            pass

    # Test
    model = load_model_with_weights(model, model_name)
    evaluate(model, test_data, test_labels, test_input_length, test_label_length, num_key)

if __name__ == "__main__":
    main()
