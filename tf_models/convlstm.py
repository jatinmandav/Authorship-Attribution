import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import array_ops as tf_array_ops

class Attention:
    def __init__(self, attention_size):
        self.attention_size = attention_size

    def layer1(self, x, hidden_states):
        x = tf_array_ops.transpose(x, [1, 0, 2])
        shape = x.get_shape()
        w1 = tf.Variable(tf.random_normal([hidden_states, self.attention_size], stddev=0.1))
        w2 = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        w3 = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))

        v = tf.tanh(tf.tensordot(x, w1, axes=1) + w2)
        vu = tf.tensordot(v, w3, axes=1)

        alphas = tf.nn.softmax(vu)

        out = tf.reduce_mean(x*tf.expand_dims(alphas, -1), 1)

        return out


class ConvLSTMModel1:
    def __init__(self, hidden_states=0, no_classes=0, attention_size=0, use_attention=False):
        self.hidden_states = hidden_states
        self.no_classes = no_classes
        self.attention = None
        if use_attention:
            self.attention = Attention(attention_size)

    def conv2d(self, x, filter, strides, padding='VALID'):
        return tf.nn.conv2d(x, filter=filter, strides=strides, padding=padding)

    def max_pool2d(self, x, ksize, strides, padding='VALID'):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    def dropout(self, x, keep_rate):
        return tf.nn.dropout(x, keep_rate)

    def relu(self, x):
        return tf.nn.relu(x)

    def model(self, x):
        weights1 = tf.Variable(tf.random_normal([3, 3, 1, 8], stddev=0.1))
        bias1 = tf.Variable(tf.random_normal([8], stddev=0.1))
        conv1 = self.conv2d(x, filter=weights1, strides=[1, 1, 1, 1])
        relu1 = self.relu(tf.add(conv1, bias1))
        max_pool1 = self.max_pool2d(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])


        weights2 = tf.Variable(tf.random_normal([5, 5, 8, 16], stddev=0.1))
        bias2 = tf.Variable(tf.random_normal([16], stddev=0.1))
        conv2 = self.conv2d(max_pool1, filter=weights2, strides=[1, 1, 1, 1])
        relu2 = self.relu(tf.add(conv2, bias2))
        max_pool2 = self.max_pool2d(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])


        weights3 = tf.Variable(tf.random_normal([7, 7, 16, 32], stddev=0.1))
        bias3 = tf.Variable(tf.random_normal([32], stddev=0.1))
        conv3 = self.conv2d(max_pool2, filter=weights3, strides=[1, 1, 1, 1])
        relu3 = self.relu(tf.add(conv3, bias3))

        weights4 = tf.Variable(tf.random_normal([11, 11, 32, 64], stddev=0.1))
        bias4 = tf.Variable(tf.random_normal([64], stddev=0.1))
        conv4 = self.conv2d(relu3, filter=weights4, strides=[1, 1, 1, 1])
        relu4 = self.relu(tf.add(conv4, bias4))

        max_pool3 = self.max_pool2d(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

        shape = max_pool3.get_shape()
        flattened = tf.reshape(max_pool3, [-1, shape[1]*shape[2], shape[3]])
        flattened = tf.transpose(flattened, [0, 2, 1])

        lstm_inp = tf.unstack(flattened, 64, 1)

        lstmcells = []
        for _ in range(2):
            lstmcells.append(rnn.BasicLSTMCell(self.hidden_states))

        multilstm= rnn.MultiRNNCell(lstmcells)
        rnn_output, states = tf.nn.static_rnn(multilstm, lstm_inp, dtype=tf.float32)

        if not self.attention == None:
            rnn_output = tf.convert_to_tensor(rnn_output)
            rnn_output = tf.tanh(rnn_output)
            att_out = self.attention.layer1(rnn_output, self.hidden_states)
            conv_lstm_out = att_out
        else:
            rnn_output = tf.tanh(rnn_output[-1])
            conv_lstm_out = rnn_output


        weight5 = tf.Variable(tf.random_normal([self.hidden_states, self.no_classes]))
        bias5 = tf.Variable(tf.random_normal([self.no_classes]))
        dense1 = tf.add(tf.matmul(conv_lstm_out, weight5), bias5)

        return dense1

class ConvLSTMModel2:
    def __init__(self, hidden_states=0, no_classes=0, attention_size=0, use_attention=False):
        self.hidden_states = hidden_states
        self.no_classes = no_classes
        self.attention = None
        if use_attention:
            self.attention = Attention(attention_size)

    def conv2d(self, x, filter, strides, padding='SAME'):
        return tf.nn.conv2d(x, filter=filter, strides=strides, padding=padding)

    def max_pool2d(self, x, ksize, strides, padding='VALID'):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    def dropout(self, x, keep_rate):
        return tf.nn.dropout(x, keep_rate)

    def relu(self, x):
        return tf.nn.relu(x)

    def model(self, x):
        weights1 = tf.Variable(tf.random_normal([2, 2, 1, 16], stddev=0.1))
        bias1 = tf.Variable(tf.random_normal([16], stddev=0.1))
        conv1 = self.conv2d(x, filter=weights1, strides=[1, 1, 1, 1])
        relu1 = self.relu(tf.add(conv1, bias1))
        max_pool1 = self.max_pool2d(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        max_pool1 = self.dropout(max_pool1, .25)

        weights2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.1))
        bias2 = tf.Variable(tf.random_normal([32], stddev=0.1))
        conv2 = self.conv2d(max_pool1, filter=weights2, strides=[1, 1, 1, 1])
        relu2 = self.relu(tf.add(conv2, bias2))

        weights3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1))
        bias3 = tf.Variable(tf.random_normal([64], stddev=0.1))
        conv3 = self.conv2d(relu2, filter=weights3, strides=[1, 1, 1, 1])
        relu3 = self.relu(tf.add(conv3, bias3))

        max_pool2 = self.max_pool2d(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        max_pool2 = self.dropout(max_pool2, .25)

        weights4 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.1))
        bias4 = tf.Variable(tf.random_normal([128], stddev=0.1))
        conv4 = self.conv2d(max_pool2, filter=weights4, strides=[1, 1, 1, 1])
        relu4 = self.relu(tf.add(conv4, bias4))

        weights5 = tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1))
        bias5 = tf.Variable(tf.random_normal([128], stddev=0.1))
        conv5 = self.conv2d(relu4, filter=weights5, strides=[1, 1, 1, 1])
        relu5 = self.relu(tf.add(conv5, bias5))

        max_pool3 = self.max_pool2d(relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        max_pool3 = self.dropout(max_pool3, .25)

        shape = max_pool3.get_shape()
        flattened = tf.reshape(max_pool3, [-1, shape[1]*shape[2], shape[3]])
        flattened = tf.transpose(flattened, [0, 2, 1])

        lstm_inp = tf.unstack(flattened, 128, 1)

        lstmcells = []
        for _ in range(2):
            lstmcells.append(rnn.BasicLSTMCell(self.hidden_states))

        multilstm= rnn.MultiRNNCell(lstmcells)
        rnn_output, states = tf.nn.static_rnn(multilstm, lstm_inp, dtype=tf.float32)

        if not self.attention == None:
            rnn_output = tf.convert_to_tensor(rnn_output)
            rnn_output = tf.tanh(rnn_output)
            att_out = self.attention.layer1(rnn_output, self.hidden_states)
            conv_lstm_out = att_out
        else:
            rnn_output = tf.tanh(rnn_output[-1])
            conv_lstm_out = rnn_output

        conv_lstm_out = self.dropout(conv_lstm_out, .10)
        weight6 = tf.Variable(tf.random_normal([self.hidden_states, self.no_classes]))
        bias6 = tf.Variable(tf.random_normal([self.no_classes]))
        dense1 = tf.add(tf.matmul(conv_lstm_out, weight6), bias6)

        return dense1
