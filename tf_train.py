from tf_models.lstm import LSTMModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from ReadData import ReadData

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help='Name of Model to use [lstm, cnn, cnnlstm]', required=True)
parser.add_argument('--training_csv', '-csv', help='Path to Training CSV file', required=True)
parser.add_argument('--embedding', '-e', help='Path to word embedding model | Default: "embeddings/skipgram-100/skipgram.bin"', default='embeddings/skipgram-100/skipgram.bin')
parser.add_argument('--pos_model', '-pos', help='Path to POS embedding model | Default: "embeddings/skipgram-pos-100/skipgram_pos.bin"', default='embeddings/skipgram-pos-100/skipgram_pos.bin')
parser.add_argument('--n_classes', '-n', help='No of classes to predict | Default: 2', default=2, type=int)
parser.add_argument('--optimizer', '-o', help='which Optimizer to use? | Default: "Adam"', default='adam')
parser.add_argument('--batch_size', '-b', help='What should be the batch size? | Default: 32', default=32, type=int)
parser.add_argument('--epochs', '-ep', help='How many epochs to Train? | Default: 100', default=100, type=int)
parser.add_argument('--train_val_split', '-s', help='What should be the train vs val split fraction? | Default: 0.1', default=0.1, type=float)
parser.add_argument('--no_samples', '-ns', help='How many samples to train on? | Default: 1000', default=1000, type=int)
parser.add_argument('--learning_rate', '-lr', help='What should be the learning rate? | Default: 0.001', default=0.001, type=float)

args = parser.parse_args()

hidden_states = 1024
classes = args.n_classes
timesteps = 75
embed_size = 101

x = tf.placeholder("float", [None, timesteps, embed_size])
y = tf.placeholder("float", [None, classes])

model = LSTMModel(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps)

reader = ReadData(args.training_csv, args.embedding, args.pos_model,
                  batch_size=args.batch_size, no_samples=args.no_samples,
                  train_val_split=args.train_val_split)

'''print('Reading Training data.')
train_x, train_y = reader.read_all_train()'''
print('Reading Validation data.')
val_x, val_y = reader.read_all_val()

prediction = model.model(x)
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(cost_func)

correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(args.epochs):
        i = 0
        epoch_loss = 0
        no_batches = int(reader.train_size/args.batch_size)
        #while i < reader.train_size:
        loss = 0
        acc = 0
        with tqdm(total=no_batches, desc="Epoch {}/{}: loss: {} acc: {}".format(epoch + 1, args.epochs, loss, acc)) as pbar:
            for _ in range(no_batches):
                start = i
                end = i + args.batch_size
                i = end

                epoch_x, epoch_y = reader.get_next_batch(start, end)
                #epoch_x, epoch_y = train_x[start:end], train_y[start:end]
                #epoch_x = np.reshape(epoch_x, [args.batch_size, len(epoch_x[0]), len(epoch_x[0][0])])
                _, c = sess.run([optimizer, cost_func], feed_dict={x: epoch_x, y:epoch_y})
                a = accuracy.eval({x: epoch_x, y: epoch_y})
                if loss == 0 and acc == 0:
                    loss = c
                    acc = a
                else:
                    loss += c
                    loss /= 2
                    acc += a
                    acc /= 2

                pbar.set_description(desc=("Epoch {}/{}: loss: {:03f}".format(epoch + 1, args.epochs, loss) + " acc: {:03f}".format(acc)))
                pbar.update(1)


            #print("Loss: {}. Accuracy: {}".format(c, accuracy.eval({x: epoch_x, y: epoch_y})))
            #epoch_loss += c

        #print("Epoch {} of {}. Loss: {}. Accuracy: {}".format(epoch + 1, args.epochs, epoch_loss, accuracy.eval({x: train_x, y: train_y})))
        print('------------------------------------------------------------')
        print("Val Loss: {} Val Accuracy: {}".format(cost_func.eval({x: val_x, y: val_y}), accuracy.eval({x: val_x, y: val_y})))
        print('------------------------------------------------------------')

    print("Accuracy: {}".format(accuracy.eval({x: val_x, y: val_y})))
