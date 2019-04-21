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

import argparse

from text2vector import Text2Vector

parser = argparse.ArgumentParser()
#parser.add_argument('--model', '-m', help='Name of Model to use [lstm, cnn, cnnlstm, bilstm, res_lstm, cnnlstmdeep]', required=True)
parser.add_argument('--text', '-t', help='Path to TEXT file that needs to be analyzed', required=True, type=str)
#parser.add_argument('--classes', '-c', help='Which model to train? ["Gender", "Age_Group", "Profession"]', required=True)
parser.add_argument('--weights', '-w', help='Path to Pre-trained model to continue training', required=True)
parser.add_argument('--embedding', '-e', help='Path to word embedding model | Default: "embeddings/skipgram-100/skipgram.bin"', default='embeddings/skipgram-100/skipgram.bin')
parser.add_argument('--n_classes', '-n', help='No of classes to predict | Default: 2', default=2, type=int)
parser.add_argument('--use_attention', '-att', help="Whether to use Attetion layer or not? | Default: False", action="store_true")
parser.add_argument('--attention_size', '-ats', help="What should be the size of attention layer? | Default: 64", default=64, type=int)
parser.add_argument('--hidden_states', '-hds', help="How many hidden states on LSTM? | Default: 128", default=128, type=int)

args = parser.parse_args()

#with open(os.path.join(args.weights, 'model.json'), 'w') as f:
#    json.dump(args.__dict__, f)

#exit()


text = open(args.text, 'r').read()
text2vec = Text2Vector(args.embedding, size=(75, 101))
vector = text2vec.convert(text)
vector = np.expand_dims(vector, axis=0)
print(vector.shape)

weights = args.weights
with open(os.path.join(args.weights, 'model.json'), 'r') as f:
    args.__dict__ = json.load(f)

args.weights = weights

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

elif args.model == 'res_lstm':
    timesteps = 75
    embed_size = 101
    hidden_states = args.hidden_states

    x = tf.placeholder("float", [None, timesteps, embed_size], name='InputData')
    y = tf.placeholder("float", [None, classes], name='Label')

    model = ResLSTM(hidden_states=hidden_states, no_classes=classes, timesteps=timesteps,
                      attention_size=attention_size, use_attention=args.use_attention, inference=True)

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

with tf.name_scope('Model'):
    prediction = model.model(x)

saver = tf.train.Saver()

with tf.Session() as sess:
    try:
        saver.restore(sess, tf.train.latest_checkpoint(args.weights))
        print()
        print('Model Successfully loaded from {}'.format(args.weights))
        print()
    except Exception as e:
        print(e)

    sess.run(tf.global_variables_initializer())
    pred = sess.run([prediction], feed_dict={x: vector})
    pred = sess.run(tf.nn.softmax(pred[0]))[0]

print('Male: {:.3f} %'.format(pred[0]), ', Female: {:.3f} %'.format(pred[1]))
