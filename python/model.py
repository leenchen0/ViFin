
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Lambda, GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as load

def create_model(feature_size, num_classes):
    input_data = Input(name='input', shape=(None, feature_size), dtype='float32')

    # LSTM layer
    lstm_outputs = GRU(32, return_sequences=True, name='lstm1')(input_data)

    # Softmax layer
    outputs = Dense(num_classes + 1, activation='softmax', name='softmax_layer')(lstm_outputs)

    labels = Input(name='labels',
                   shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(
        ctc_lambda_func, output_shape=(1,),
        name='ctc')([outputs, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    optimizer = Adam()

    model = Model(inputs=[input_data, labels, input_length, label_length],
                  outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    model.summary()

    return model

def load_raw_lstm_ctc_model(model_path):
    return load(model_path, { 'keras': keras, '<lambda>': lambda y_true, y_pred: y_pred })

def load_model_with_weights(model, weights_path):
    model.load_weights(weights_path)
    return Model(inputs=model.input[0], outputs=model.get_layer('softmax_layer').output)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def load_model(model_path):
    return load(model_path)
