import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class Attention:
    def __init__(self, attention_size):
        self.attention_size = attention_size

    def layer2(self, x, hidden_size):
        Q = tf.Variable(tf.random_normal([hidden_size, self.attention_size], stddev=0.1))
        Q = tf.matmul(x, Q)

        K = tf.Variable(tf.random_normal([hidden_size, self.attention_size], stddev=0.1))
        K = tf.matmul(x, K)

        attention = tf.matmul(Q, K, transpose_b=True)

        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))

        attention = tf.nn.softmax(attention, dim=-1)

        V = tf.Variable(tf.random_normal([hidden_size, self.attention_size], stddev=0.1))
        V = tf.matmul(x, V)

        output = tf.matmul(attention, V)
        return output

    def layer3(self, x, timesteps, hidden_size):
        Q = tf.Variable(tf.random_normal([timesteps, hidden_size, self.attention_size], stddev=0.1))
        Q = tf.matmul(x, Q)

        K = tf.Variable(tf.random_normal([timesteps, hidden_size, self.attention_size], stddev=0.1))
        K = tf.matmul(x, K)

        attention = tf.matmul(Q, K, transpose_b=True)

        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))

        attention = tf.nn.softmax(attention, dim=-1)

        V = tf.Variable(tf.random_normal([timesteps, hidden_size, self.attention_size], stddev=0.1))
        V = tf.matmul(x, V)

        output = tf.matmul(attention, V)
        return output


    def layer1(self, x, hidden_size):
        w_omega = tf.Variable(tf.random_normal([hidden_size, self.attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))

        v = tf.tanh(tf.add(tf.tensordot(x, w_omega, axes=1), b_omega))
        vu = tf.add(tf.tensordot(v, u_omega, axes=1), b_omega)

        alphas = tf.nn.softmax(vu)

        output = tf.reduce_sum(tf.multiply(x, tf.expand_dims(alphas, -1)), 1)

        return output

class AttentionLSTMModel:
    def __init__(self, hidden_states=0, no_classes=0, timesteps=0, attention_size=0):
        self.hidden_states = hidden_states
        self.no_classes = no_classes
        self.timesteps = timesteps
        self.attention = Attention(attention_size)

    def model(self, x):
        x = tf.unstack(x, self.timesteps, 1)

        #lstmcells = []
        #for _ in range(2):
        #    lstmcells.append(rnn.BasicLSTMCell(self.hidden_states))

        lstm = rnn.BasicLSTMCell(self.hidden_states)

        #multilstm = rnn.MultiRNNCell(lstmcells)
        rnn_output, states = tf.nn.static_rnn(lstm, x, dtype=tf.float32)

        #rnn_output = tf.reshape(rnn_output, [-1, len(rnn_output), rnn_output[-1].shape[1]])

        #print(states.shape)
        #print(states.get_shape())
        #print(rnn_output.get_shape())

        rnn_output = rnn_output[-1]
        #rnn_output = np.array(rnn_output)

        #attention = self.attention.layer3(rnn_output, self.timesteps, self.hidden_states)
        attention = self.attention.layer2(rnn_output, self.hidden_states)
        #attention = tf.reshape(attention, [-1, self.timesteps*self.attention.attention_size])

        weights1 = tf.Variable(tf.random_normal([self.attention.attention_size, 512]))
        biases1 = tf.Variable(tf.random_normal([512]))
        output1 = tf.add(tf.matmul(attention, weights1), biases1)
        output1 = tf.nn.relu(output1)

        output1 = tf.nn.dropout(output1, 0.75)

        weights2 = tf.Variable(tf.random_normal([512, self.no_classes]))
        biases2 = tf.Variable(tf.random_normal([self.no_classes]))
        output2 = tf.add(tf.matmul(output1, weights2), biases2)

        return output2

if __name__ == "__main__":
    hidden_states = 512
    classes = 2
    timesteps = 1
    embed_size = 100
    x = tf.placeholder("float", [None, timesteps, embed_size])
    y = tf.placeholder("float", [None, classes])

    model = LSTMModel(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps)
