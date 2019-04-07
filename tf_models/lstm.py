import tensorflow as tf
from tensorflow.contrib import rnn

class LSTMModel:
    def __init__(self, hidden_states=0, no_classes=0, timesteps=0):
        self.hidden_states = hidden_states
        self.no_classes = no_classes
        self.timesteps = timesteps

    def model(self, x):
        x = tf.unstack(x, self.timesteps, 1)

        lstmcells = []
        for _ in range(3):
            lstmcells.append(rnn.BasicLSTMCell(self.hidden_states))

        multilstm= rnn.MultiRNNCell(lstmcells)
        rnn_output, states = tf.nn.static_rnn(multilstm, x, dtype=tf.float32)

        weights = tf.Variable(tf.random_normal([self.hidden_states, self.no_classes]))
        biases = tf.Variable(tf.random_normal([self.no_classes]))

        output = tf.add(tf.matmul(rnn_output[-1], weights), biases)

        return output

if __name__ == "__main__":
    hidden_states = 512
    classes = 2
    timesteps = 1
    embed_size = 100
    x = tf.placeholder("float", [None, timesteps, embed_size])
    y = tf.placeholder("float", [None, classes])

    model = LSTMModel(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps)
