from tf_models.lstm import LSTMModel
from tf_models.convlstm import ConvLSTMModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook, tqdm
import os

from ReadData import ReadData

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help='Name of Model to use [lstm, cnn, cnnlstm]', required=True)
parser.add_argument('--training_csv', '-csv', help='Path to Training CSV file', required=True)
parser.add_argument('--embedding', '-e', help='Path to word embedding model | Default: "embeddings/skipgram-100/skipgram.bin"', default='embeddings/skipgram-100/skipgram.bin')
parser.add_argument('--n_classes', '-n', help='No of classes to predict | Default: 2', default=2, type=int)
parser.add_argument('--optimizer', '-o', help='which Optimizer to use? | Default: "Adam"', default='adam')
parser.add_argument('--batch_size', '-b', help='What should be the batch size? | Default: 32', default=32, type=int)
parser.add_argument('--epochs', '-ep', help='How many epochs to Train? | Default: 100', default=100, type=int)
parser.add_argument('--train_val_split', '-s', help='What should be the train vs val split fraction? | Default: 0.1', default=0.1, type=float)
parser.add_argument('--no_samples', '-ns', help='How many samples to train on? | Default: 1000', default=1000, type=int)
parser.add_argument('--learning_rate', '-lr', help='What should be the learning rate? | Default: 0.001', default=0.001, type=float)
parser.add_argument('--logs', '-l', help="Where should the trained model be saved? | Default: logs", default='logs')

args = parser.parse_args()

hidden_states = 1024
classes = args.n_classes
timesteps = 75
embed_size = 101

if args.model == 'lstm':
    x = tf.placeholder("float", [None, timesteps, embed_size], name='InputData')
    y = tf.placeholder("float", [None, classes], name='Label')

    model = LSTMModel(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps)
elif args.model.startswith('cnn'):
    x = tf.placeholder("float", [None, timesteps, embed_size, 1], name='InputData')
    y = tf.placeholder("float", [None, classes], name='Label')
    model = ConvLSTMModel()

reader = ReadData(args.training_csv, args.embedding,
                  batch_size=args.batch_size, no_samples=args.no_samples,
                  train_val_split=args.train_val_split)

print('Reading Validation data.')
val_x, val_y = reader.read_all_val()
if args.model.startswith('cnn'):
    val_x = np.reshape(val_x, (val_x.shape[0], timesteps, embed_size, 1))

with tf.name_scope('Model'):
    prediction = model.model(x)

with tf.name_scope('Loss'):
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(cost_func)

with tf.name_scope('Accuracy'):
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

if not os.path.exists(args.logs):
    os.mkdir(args.logs)

saver = tf.train.Saver()
weights_path = os.path.join(args.logs, 'weights')
if not os.path.exists(weights_path):
    os.mkdir(weights_path)

tensorboard_path = os.path.join(args.logs, 'tensorboard')
if not os.path.exists(tensorboard_path):
    os.mkdir(tensorboard_path)

train_log = os.path.join(tensorboard_path, 'training')
val_log = os.path.join(tensorboard_path, 'validation')

tf.summary.scalar('loss', cost_func)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()

prev_val_loss = float('inf')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_summary_writer = tf.summary.FileWriter(train_log, graph=sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log)

    for epoch in range(args.epochs):
        i = 0
        epoch_loss = 0
        no_batches = int(reader.train_size/args.batch_size)

        loss = []
        acc = []
        with tqdm(total=no_batches, desc="Epoch {}/{}: loss: {} acc: {}".format(epoch + 1, args.epochs, loss, acc)) as pbar:
            for batch_num in range(no_batches):
                start = i
                end = i + args.batch_size
                i = end

                epoch_x, epoch_y = reader.get_next_batch(start, end)
                if args.model.startswith('cnn'):
                    epoch_x = np.reshape(epoch_x, (epoch_x.shape[0], timesteps, embed_size, 1))
                _, c, summary = sess.run([optimizer, cost_func, merged_summary_op], feed_dict={x: epoch_x, y:epoch_y})
                train_summary_writer.add_summary(summary, epoch*no_batches+batch_num)

                a = accuracy.eval({x: epoch_x, y: epoch_y})
                loss.append(c)
                acc.append(a)

                pbar.set_description(desc=("Epoch {}/{}: loss: {:.03f}".format(epoch + 1, args.epochs, np.average(loss)) + " acc: {:.03f}".format(np.average(acc))))
                pbar.update(1)

        print('------------------------------------------------------------')
        val_loss, val_acc, val_summary = sess.run([cost_func, accuracy, merged_summary_op], feed_dict={x: epoch_x, y:epoch_y})

        val_summary_writer.add_summary(val_summary, epoch)

        val_loss = cost_func.eval({x: val_x, y: val_y})
        val_acc = accuracy.eval({x: val_x, y: val_y})
        print("Val Loss: {} Val Accuracy: {}".format(val_loss, val_acc))
        print('------------------------------------------------------------')

        if val_loss < prev_val_loss:
            prev_val_loss = val_loss
            model_name = 'ep{:03d}'.format(epoch+1) + '-loss{:.03f}'.format( np.average(loss)) + '-val_loss{:.03f}.ckpt'.format(val_loss)
            saver.save(sess, os.path.join(weights_path, model_name))

    print("Accuracy: {}".format(accuracy.eval({x: val_x, y: val_y})))
