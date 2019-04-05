from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Activation, RepeatVector, TimeDistributed
from keras.layers.normalization import BatchNormalization

import keras.backend as K

def LSTMModel(input_shape, output_shape):
    inp = Input(shape=input_shape)

    x = LSTM(128)(inp)
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(128)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    out = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model

if __name__ == "__main__":
    model = LSTMModel((100,1), 6)
    model.summary()
