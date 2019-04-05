from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Activation, RepeatVector, TimeDistributed
from keras.layers.normalization import BatchNormalization

import keras.backend as K

def CNNModel(input_shape, output_shape):
    inp = Input(shape=input_shape)

    x = Conv1D(8, kernel_size=3, padding='same', activation='relu')(inp)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    out = Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model


if __name__ == "__main__":
    model = CNNModel((100,1), 6)
    model.summary()
