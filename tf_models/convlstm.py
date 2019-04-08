import tensorflow as tf
from tensorflow.contrib import rnn

class ConvLSTMModel:
    def __init__(self, hidden_states=0, no_classes=0):
        self.hidden_states = hidden_states
        self.no_classes = no_classes

    def conv2d(self, x, filter, strides, padding='SAME'):
        return tf.nn.conv2d(x, filter=filter, strides=strides, padding=padding)

    def max_pool2d(self, x, ksize, strides, padding='SAME'):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    def dropout(self, x, keep_rate):
        return tf.nn.dropout(x, keep_rate)

    def relu(self, x):
        return tf.nn.relu(x)

    def model(self, x):
        weight1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
        bias1 = tf.Variable(tf.random_normal([32]))
        conv1 = self.conv2d(x, filter=weight1, strides=[1, 2, 2, 1])
        relu1 = self.relu(tf.add(conv1, bias1))
        max_pool1 = self.max_pool2d(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        weight2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
        bias2 = tf.Variable(tf.random_normal([64]))
        conv2 = self.conv2d(max_pool1, filter=weight2, strides=[1, 2, 2, 1])
        relu2 = self.relu(tf.add(conv2, bias2))

        weight3 = tf.Variable(tf.random_normal([3, 3, 64, 64]))
        bias3 = tf.Variable(tf.random_normal([64]))
        conv3 = self.conv2d(relu2, filter=weight3, strides=[1, 2, 2, 1])
        relu3 = self.relu(tf.add(conv3, bias3))

        max_pool2 = self.max_pool2d(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        shape = max_pool2.get_shape()
        print(shape)
        '''shape = max_pool2.get_shape
        print(shape)
        shape = tf.shape(max_pool2)
        print(shape)'''
        lstm_inp = tf.reshape(max_pool2, [-1, shape[1]*shape[2], shape[3]])

        lstm_inp = tf.unstack(lstm_inp, shape[1]*shape[2], 1)

        lstmcells = []
        for _ in range(2):
            lstmcells.append(rnn.BasicLSTMCell(self.hidden_states))

        multilstm= rnn.MultiRNNCell(lstmcells)
        rnn_output, states = tf.nn.static_rnn(multilstm, lstm_inp, dtype=tf.float32)
        rnn_output = tf.nn.relu(rnn_output[-1])

        weight4 = tf.Variable(tf.random_normal([self.hidden_states, 512]))
        bias4 = tf.Variable(tf.random_normal([512]))
        dense1 = tf.add(tf.matmul(rnn_output, weight4), bias4)

        dropout1 = self.dropout(dense1, 0.8)

        weight5 = tf.Variable(tf.random_normal([512, 2]))
        bias5 = tf.Variable(tf.random_normal([2]))
        dense2 = tf.add(tf.matmul(dropout1, weight5), bias5)

        return dense2
