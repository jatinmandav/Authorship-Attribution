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
        for _ in range(1):
            lstmcells.append(rnn.BasicLSTMCell(self.hidden_states))

        multilstm= rnn.MultiRNNCell(lstmcells)
        rnn_output, states = tf.nn.static_rnn(multilstm, x, dtype=tf.float32)

        #rnn_output = tf.nn.relu(rnn_output[-1])
        print(rnn_output[0].get_shape())

        weights1 = tf.Variable(tf.random_normal([self.timesteps, self.hidden_states]))
        #biases1 = tf.Variable(tf.random_normal([512]))
        #output1 = tf.add(tf.matmul(rnn_output, weights1), biases1)
        output1 = tf.matmul(rnn_output, weights1)
        output1 = tf.nn.relu(output1)

        output1 = tf.nn.dropout(rnn_output, 0.75)

        weights2 = tf.Variable(tf.random_normal([self.hidden_states, self.no_classes]))
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
