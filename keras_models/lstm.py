from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Activation, RepeatVector, TimeDistributed
from keras.layers.normalization import BatchNormalization

import keras.backend as K

def LSTMModel(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(512))
    model.add(Dense(output_shape, activation='softmax'))

    return model

if __name__ == "__main__":
    model = LSTMModel((100,101), 2)
    model.summary()
