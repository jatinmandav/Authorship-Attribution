from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Activation, RepeatVector, TimeDistributed
from keras.layers.normalization import BatchNormalization

import keras.backend as K

def CNNLSTMModel(input_shape, output_shape):
    inp = Input(shape=input_shape)

    x = Conv1D(32, kernel_size=5, padding='same', activation='relu')(inp)
    #x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    #x = Conv1D(128, kernel_size=7, padding='same', activation='relu')(x)
    #x = MaxPooling1D(pool_size=5, strides=2)(x)

    #x = LSTM(256, return_sequences=True)(x)

    x = Flatten()(x)
    '''x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
'''
    out = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model

if __name__ == "__main__":
    model = CNNLSTMModel((75,101), 2)
    model.summary()
