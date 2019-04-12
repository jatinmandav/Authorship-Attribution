from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Activation, RepeatVector, TimeDistributed
from keras.layers.normalization import BatchNormalization

import keras.backend as K

def LSTMModel(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(256))


    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='softmax'))

    return model

if __name__ == "__main__":
    model = LSTMModel((75,101), 2)
    model.summary()
