import tensorflow as tf

class ConvLSTMModel:
    def __init__(self):
        pass

    def conv2d(self, x, filter, strides, padding='SAME'):
        return tf.nn.conv2d(x, filter=filter, strides=strides, padding=padding)

    def max_pool2d(self, x, ksize, strides, padding='SAME'):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)

    def dropout(self, x, keep_rate):
        return tf.nn.dropout(x, keep_rate)

    def relu(self, x):
        return tf.nn.relu(x)

    def batch_normalization(self, x):
        return tf.nn.batch_normalization(x)

    def model(self, x):
        weight1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
        bias1 = tf.Variable(tf.random_normal([32]))
        conv1 = self.conv2d(x, filter=weight1, strides=[1, 2, 2, 1])
        relu1 = self.relu(tf.add(conv1, bias1))
        max_pool1 = self.max_pool2d(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        #max_pool1 = self.batch_normalization(max_pool1)

        weight2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
        bias2 = tf.Variable(tf.random_normal([64]))
        conv2 = self.conv2d(max_pool1, filter=weight2, strides=[1, 2, 2, 1])
        relu2 = self.relu(tf.add(conv2, bias2))
        max_pool2 = self.max_pool2d(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        #max_pool2 = self.batch_normalization(max_pool2)

        weight3 = tf.Variable(tf.random_normal([3, 3, 64, 128]))
        bias3 = tf.Variable(tf.random_normal([128]))
        conv3 = self.conv2d(max_pool2, filter=weight3, strides=[1, 2, 2, 1])
        relu3 = self.relu(tf.add(conv3, bias3))
        max_pool3 = self.max_pool2d(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        #max_pool3 = self.batch_normalization(max_pool3)

        flatten = tf.reshape(max_pool3, [-1, 2*2*128])

        weight4 = tf.Variable(tf.random_normal([2*2*128, 512]))
        bias4 = tf.Variable(tf.random_normal([512]))
        dense1 = tf.add(tf.matmul(flatten, weight4), bias4)

        dropout1 = self.dropout(dense1, 0.8)

        weight5 = tf.Variable(tf.random_normal([512, 2]))
        bias5 = tf.Variable(tf.random_normal([2]))
        dense2 = tf.add(tf.matmul(dropout1, weight5), bias5)

        return dense2
