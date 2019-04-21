from tf_models.lstm import LSTMModel
from tf_models.conv import ConvModel
from tf_models.convlstm import ConvLSTMModel1, ConvLSTMModel2
from tf_models.bidirectional_lstm import BiLSTMModel

from tf_models.res_lstm import ResLSTM

import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook, tqdm
import os
import json

import tensorflow.contrib.slim as slim

from ReadData import ReadData

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', help='Name of Model to use [lstm, cnn, cnnlstm, bilstm, res_lstm, cnnlstmdeep]', required=True)
parser.add_argument('--training_csv', '-csv', help='Path to Training CSV file', required=True)
parser.add_argument('--classes', '-c', help='Which model to train? ["Gender", "Age_Group", "Profession"]', required=True)
parser.add_argument('--embedding', '-e', help='Path to word embedding model | Default: "embeddings/skipgram-100/skipgram.bin"', default='embeddings/skipgram-100/skipgram.bin')
parser.add_argument('--weights', '-w', help='Path to Pre-trained model to continue training')
parser.add_argument('--n_classes', '-n', help='No of classes to predict | Default: 2', default=2, type=int)
parser.add_argument('--optimizer', '-o', help='which Optimizer to use? | Default: "Adam"', default='adam')
parser.add_argument('--batch_size', '-b', help='What should be the batch size? | Default: 32', default=32, type=int)
parser.add_argument('--epochs', '-ep', help='How many epochs to Train? | Default: 5', default=5, type=int)
parser.add_argument('--initial_epoch', '-iep', help='Where to continue from? | Default: 0', default=0, type=int)
parser.add_argument('--steps', '-st', help='How many steps to Train? | Default: 100000', default=100000, type=int)
parser.add_argument('--train_val_split', '-s', help='What should be the train vs val split fraction? | Default: 0.1', default=0.1, type=float)
parser.add_argument('--no_samples', '-ns', help='How many samples to train on? | Default: 1000', default=1000, type=int)
parser.add_argument('--learning_rate', '-lr', help='What should be the learning rate? | Default: 0.00001', default=0.00001, type=float)
parser.add_argument('--lr_change', '-clr', help='How often should the learning rate be increased? | Default: 10000', default=10000, type=int)
parser.add_argument('--logs', '-l', help="Where should the trained model be saved? | Default: logs", default='logs')
parser.add_argument('--data_overlap', '-ol', help="What percent of data should overlap with each batch? | Default: 0.2", default=0.2, type=float)
parser.add_argument('--use_attention', '-att', help="Whether to use Attetion layer or not? | Default: False", action="store_true")
parser.add_argument('--attention_size', '-ats', help="What should be the size of attention layer? | Default: 64", default=64, type=int)
parser.add_argument('--hidden_states', '-hds', help="How many hidden states on LSTM? | Default: 128", default=128, type=int)

args = parser.parse_args()

classes = args.n_classes

attention_size = args.attention_size

if args.model == 'lstm':
    timesteps = 75
    embed_size = 101
    hidden_states = args.hidden_states

    x = tf.placeholder("float", [None, timesteps, embed_size], name='InputData')
    y = tf.placeholder("float", [None, classes], name='Label')

    model = LSTMModel(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps,
                      attention_size=attention_size, use_attention=args.use_attention)

elif args.model == 'bilstm':
    timesteps = 75
    embed_size = 101
    hidden_states = args.hidden_states

    x = tf.placeholder("float", [None, timesteps, embed_size], name='InputData')
    y = tf.placeholder("float", [None, classes], name='Label')

    model = BiLSTMModel(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps,
                        attention_size=attention_size, use_attention=args.use_attention)

if args.model == 'res_lstm':
    timesteps = 75
    embed_size = 101
    hidden_states = args.hidden_states

    x = tf.placeholder("float", [None, timesteps, embed_size], name='InputData')
    y = tf.placeholder("float", [None, classes], name='Label')

    model = ResLSTM(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps,
                      attention_size=attention_size, use_attention=args.use_attention)

elif args.model.startswith('cnn'):
    timesteps = 75
    embed_size = 101
    hidden_states = args.hidden_states

    x = tf.placeholder("float", [None, timesteps, embed_size, 1], name='InputData')
    y = tf.placeholder("float", [None, classes], name='Label')

    if args.model.endswith('lstm'):
        model = ConvLSTMModel1(hidden_states, classes, attention_size=attention_size,
                              use_attention=args.use_attention)
    elif args.model.endswith('deep'):
          model = ConvLSTMModel2(hidden_states, classes, attention_size=attention_size,
                                use_attention=args.use_attention)
    else:
        model = ConvModel(classes)

reader = ReadData(args.training_csv, args.embedding, args.classes,
                  batch_size=args.batch_size, no_samples=args.no_samples,
                  train_val_split=args.train_val_split)

print('Reading Validation data.')
val_x, val_y = reader.read_all_val()
if args.model.startswith('cnn'):
    val_x = np.reshape(val_x, (val_x.shape[0], timesteps, embed_size, 1))

with tf.name_scope('Model'):
    prediction = model.model(x)

with tf.name_scope('Loss'):
    crossent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y)
    cost_func = (tf.reduce_mean(crossent))/args.batch_size
    #cost_func = tf.reduce_mean(crossent)

lr = tf.placeholder('float', [])
learning_rate = args.learning_rate

with tf.name_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost_func)
    #optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_func)

with tf.name_scope('Accuracy'):
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

if args.weights == None:
    log_dir = args.logs + '_' + args.model + '_' + args.classes
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    weights_path = os.path.join(log_dir, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    tensorboard_path = os.path.join(log_dir, 'tensorboard')
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)

    train_log = os.path.join(tensorboard_path, 'training')
    val_log = os.path.join(tensorboard_path, 'validation')
else:
    log_dir = args.weights
    weights_path = os.path.join(log_dir, 'weights')
    tensorboard_path = os.path.join(log_dir, 'tensorboard')
    train_log = os.path.join(tensorboard_path, 'training')
    val_log = os.path.join(tensorboard_path, 'validation')

with open(os.path.join(weights_path, 'model.json'), 'w') as f:
    json.dump(args.__dict__, f)

saver = tf.train.Saver()

tf.summary.scalar('loss', cost_func)
tf.summary.scalar('accuracy', accuracy)
merged_summary_op = tf.summary.merge_all()

model_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(model_vars, print_info=True)

prev_val_loss = float('inf')

print('Training on {} Training samples and {} Validation samples'.format(reader.train_size, reader.val_size))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, tf.train.latest_checkpoint(weights_path))
        print()
        print('Model Successfully loaded from {}'.format(weights_path))
        print()
    except Exception as e:
        print(e)

    exit()
    train_summary_writer = tf.summary.FileWriter(train_log, graph=sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log)

    for epoch in range(args.epochs):
        i = 0
        epoch_loss = 0
        no_batches = int(reader.train_size/args.batch_size)

        loss = []
        acc = []
        updation_step = args.lr_change*2
        with tqdm(total=no_batches, desc="Epoch {}/{}: loss: {} acc: {}".format(epoch + 1, args.epochs, loss, acc)) as pbar:
            for batch_num in range(no_batches):
                start = i
                end = i + args.batch_size
                i = start + int(args.batch_size*(1-args.data_overlap))

                step = epoch*no_batches+batch_num

                epoch_x, epoch_y = reader.get_next_batch(start, end)
                if args.model.startswith('cnn'):
                    epoch_x = np.reshape(epoch_x, (epoch_x.shape[0], timesteps, embed_size, 1))

                _, c, train_summary = sess.run([optimizer, cost_func, merged_summary_op], feed_dict={lr: args.learning_rate, x: epoch_x, y:epoch_y})
                train_summary_writer.add_summary(train_summary, step)

                val_loss, val_acc, val_summary = sess.run([cost_func, accuracy, merged_summary_op], feed_dict={x: val_x, y:val_y})
                val_summary_writer.add_summary(val_summary, step)

                if step > updation_step:
                    updation_step += args.lr_change
                    if learning_rate < 1.0:
                        learning_rate = learning_rate*2.5
                        print('LR: ', learning_rate)

                a = accuracy.eval({x: epoch_x, y: epoch_y})
                loss.append(c)
                acc.append(a)

                pbar.set_description(desc=("Epoch {}/{}: loss: {:.03f}".format(epoch + 1, args.epochs, np.average(loss)) + " acc: {:.03f}".format(np.average(acc))))
                pbar.update(1)

        print('------------------------------------------------------------')
        #val_loss, val_acc, val_summary = sess.run([cost_func, accuracy, merged_summary_op], feed_dict={x: val_x, y:val_y})
        #val_summary_writer.add_summary(val_summary, epoch)

        val_loss = cost_func.eval({x: val_x, y: val_y})
        val_acc = accuracy.eval({x: val_x, y: val_y})
        print("Val Loss: {} Val Accuracy: {}".format(val_loss, val_acc))
        print('------------------------------------------------------------')

        if val_loss < prev_val_loss:
            prev_val_loss = val_loss
            model_name = 'ep{:03d}'.format(args.initial_epoch + epoch+1) + '-loss{:.03f}'.format(np.average(loss)) + '-val_loss{:.03f}.ckpt'.format(val_loss)
            saver.save(sess, os.path.join(weights_path, model_name))

    print("Accuracy: {}".format(accuracy.eval({x: val_x, y: val_y})))
