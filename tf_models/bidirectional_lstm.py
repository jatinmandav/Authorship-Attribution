import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

class Attention:
    def __init__(self):
        pass

    def layer1(self, x, hidden_size, timesteps):
        fw_out, bw_out = x

        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        H = fw_out + bw_out
        M = tf.tanh(H)

        alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, hidden_size]),
                                                         tf.reshape(W, [-1, 1])),
                                                (-1, timesteps)))

        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(alpha, [-1, timesteps, 1]))

        r = tf.squeeze(r)
        h_star = tf.tanh(r)

        return h_star


class BiLSTMModel:
    def __init__(self, hidden_states=0, no_classes=0, timesteps=0):
        self.hidden_states = hidden_states
        self.no_classes = no_classes
        self.timesteps = timesteps
        self.attention = Attention()

    def model(self, x):
        x = tf.unstack(x, self.timesteps, 1)

        forward_lstm = rnn.BasicLSTMCell(self.hidden_states)
        backward_lstm = rnn.BasicLSTMCell(self.hidden_states)

        rnn_output, f_states, b_states = tf.nn.static_bidirectional_rnn(forward_lstm, backward_lstm, x, dtype=tf.float32)
        rnn_output = tf.nn.tanh(rnn_output[-1])

        #attention = self.attention.layer1([f_states, b_states], self.hidden_states, self.timesteps)

        '''weights1 = tf.Variable(tf.random_normal([2*self.hidden_states, 512]))
        biases1 = tf.Variable(tf.random_normal([512]))
        output1 = tf.add(tf.matmul(rnn_output, weights1), biases1)
        output1 = tf.nn.relu(output1)'''

        output1 = tf.nn.dropout(rnn_output, 0.75)

        weights2 = tf.Variable(tf.random_normal([2*self.hidden_states, self.no_classes]))
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

    model = BiLSTMModel(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps)
    model = model.model(x)
