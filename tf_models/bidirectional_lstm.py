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

class BiLSTMModel:
    def __init__(self, hidden_states=0, no_classes=0, timesteps=0, attention_size=0, use_attention=False):
        self.hidden_states = hidden_states
        self.no_classes = no_classes
        self.timesteps = timesteps
        self.attention = None

        if use_attention:
            self.attention = Attention(attention_size)

    def model(self, x):
        x = tf.unstack(x, self.timesteps, 1)

        forward_lstm = rnn.BasicLSTMCell(self.hidden_states)
        backward_lstm = rnn.BasicLSTMCell(self.hidden_states)

        rnn_output, f_states, b_states = tf.nn.static_bidirectional_rnn(forward_lstm, backward_lstm, x, dtype=tf.float32)
        #rnn_output = tf.nn.tanh(rnn_output[-1])

        if not self.attention == None:
            rnn_output = tf.convert_to_tensor(rnn_output)
            rnn_output = tf.tanh(rnn_output)
            attention = self.attention.layer1(rnn_output, 2*self.hidden_states)
            bilstm_out = attention
        else:
            rnn_output = tf.tanh(rnn_output[-1])
            bilstm_out = rnn_output

        output1 = tf.nn.dropout(bilstm_out, 0.5)

        weights2 = tf.Variable(tf.random_normal([2*self.hidden_states, self.no_classes]))
        biases2 = tf.Variable(tf.random_normal([self.no_classes]))
        output2 = tf.add(tf.matmul(output1, weights2), biases2)

        return output2
