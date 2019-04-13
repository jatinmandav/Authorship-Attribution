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

class ResLSTM:
    def __init__(self, hidden_states=0, no_classes=0, timesteps=0, attention_size=0, use_attention=False):
        self.hidden_states = hidden_states
        self.no_classes = no_classes
        self.timesteps = timesteps
        self.attention = None

        if use_attention:
            self.attention = Attention(attention_size)

    def residual_block(self, x, block_num, hidden_1, hidden_2):
        x = tf.unstack(x, self.timesteps, 1)
        print(type(x), type(x[0]))
        print(x[0].get_shape())

        with tf.variable_scope('fw_lstm{}'.format(block_num), reuse=True):
            forward_lstm1 = rnn.BasicLSTMCell(hidden_1, name='fw_lstm{}'.format(block_num))
        with tf.variable_scope('bw_lstm{}'.format(block_num), reuse=True):
            backward_lstm1 = rnn.BasicLSTMCell(hidden_1, name='bw_lstm{}'.format(block_num))

        rnn_output1, f_states, b_states = tf.nn.static_bidirectional_rnn(forward_lstm1, backward_lstm1,
                                                                         x, dtype=tf.float32)

        res1 = tf_array_ops.transpose(rnn_output1, [1, 0, 2])
        print(type(rnn_output1), type(rnn_output1[0]))
        print(rnn_output1[0].get_shape())

        with tf.variable_scope('fw_lstm{}'.format(block_num+1), reuse=True):
            forward_lstm2 = rnn.BasicLSTMCell(hidden_2, name='fw_lstm{}'.format(block_num+1))
        with tf.variable_scope('bw_lstm{}'.format(block_num+1), reuse=True):
            backward_lstm2 = rnn.BasicLSTMCell(hidden_2, name='bw_lstm{}'.format(block_num+1))

        rnn_output2, f_states, b_states = tf.nn.static_bidirectional_rnn(forward_lstm2, backward_lstm2,
                                                                         rnn_output1, dtype=tf.float32)

        res2 = tf_array_ops.transpose(rnn_output2, [1, 0, 2])
        res_link = tf.concat([res1, res2], 2)

        return res1, res2, res_link

    def model(self, x):
        res1, res2, res_link1 = self.residual_block(x, 1, 16, 32)
        print('-----', res1.get_shape(), res2.get_shape(), res_link1.get_shape())
        res3, res4, res_link2 = self.residual_block(res_link1, 3, 32, 64)
        print('-----', res3.get_shape(), res4.get_shape(), res_link2.get_shape())

        res_link3 = tf.concat([res_link1, res_link2], 2)
        print(res_link3)

        res_link3 = tf.unstack(res_link3, self.timesteps, 1)

        with tf.variable_scope('fw_lstm{}'.format(5), reuse=True):
            forward_lstm = rnn.BasicLSTMCell(self.hidden_states, name='fw_lstm{}'.format(5))
        with tf.variable_scope('bw_lstm{}'.format(5), reuse=True):
            backward_lstm = rnn.BasicLSTMCell(self.hidden_states, name='bw_lstm{}'.format(5))

        reslstm_out, f_states, b_states = tf.nn.static_bidirectional_rnn(forward_lstm, backward_lstm,
                                                                         res_link3, dtype=tf.float32)

        if not self.attention == None:
            reslstm_out = tf.convert_to_tensor(reslstm_out)
            reslstm_out = tf.tanh(reslstm_out)
            attention = self.attention.layer1(reslstm_out, 2*self.hidden_states)
            print(attention.get_shape())
            bilstm_out = attention
        else:
            reslstm_out = tf.tanh(reslstm_out[-1])
            reslstm_out = reslstm_out

        output1 = tf.nn.dropout(reslstm_out, 0.5)

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
