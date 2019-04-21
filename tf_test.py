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
import pandas as pd

import tensorflow.contrib.slim as slim
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

import argparse

from text2vector import Text2Vector
from ReadData import ReadData

parser = argparse.ArgumentParser()
#parser.add_argument('--model', '-m', help='Name of Model to use [lstm, cnn, cnnlstm, bilstm, res_lstm, cnnlstmdeep]', required=True)
parser.add_argument('--test_csv', '-csv', help='Path to Testing CSV file', required=True)
parser.add_argument('--batch_size', '-b', help='What should be the batch size? | Default: 32', default=32, type=int)
parser.add_argument('--no_samples', '-ns', help='How many samples to test on? | Default: 1000', default=1000, type=int)
#parser.add_argument('--classes', '-c', help='Which model to train? ["Gender", "Age_Group", "Profession"]', required=True)
parser.add_argument('--weights', '-w', help='Path to Pre-trained model to continue training', required=True)
parser.add_argument('--embedding', '-e', help='Path to word embedding model | Default: "embeddings/skipgram-100/skipgram.bin"', default='embeddings/skipgram-100/skipgram.bin')
parser.add_argument('--n_classes', '-n', help='No of classes to predict | Default: 2', default=2, type=int)
parser.add_argument('--use_attention', '-att', help="Whether to use Attetion layer or not? | Default: False", action="store_true")
parser.add_argument('--attention_size', '-ats', help="What should be the size of attention layer? | Default: 64", default=64, type=int)
parser.add_argument('--hidden_states', '-hds', help="How many hidden states on LSTM? | Default: 128", default=128, type=int)

args = parser.parse_args()

text2vec = Text2Vector(args.embedding, size=(75, 101))
def get_embedding(text):
    vector = text2vec.convert(text)
    vector = np.expand_dims(vector, axis=0)
    return vector

weights = args.weights
no_samples = args.no_samples
with open(os.path.join(args.weights, 'model.json'), 'r') as f:
    args.__dict__.update(json.load(f))

args.weights = weights
args.no_samples = no_samples
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

with tf.name_scope('Model'):
    prediction = model.model(x)

saver = tf.train.Saver()

data = pd.read_csv(args.test_csv, sep='|')
data = data.sample(frac=1).reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True).head(args.no_samples)
data_size = len(data)

def get_classes():
    return [class_ for class_ in open(args.classes + '.txt', 'r').read().split('\n') if len(class_) > 1]

classes = get_classes()
print()
print('Testing model on following classes: ')
print(classes)
print()

label = []
predi = []
accuracy = []

predicts = []

for i in tqdm(range(data_size)):
    with tf.Session() as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(args.weights))
        except Exception as e:
            print(e)
            exit()
        sess.run(tf.global_variables_initializer())
        vector = get_embedding(data['Post'][i])
        y = classes.index(str(data[args.classes][i]))
        label.append(y)

        pred = sess.run([prediction], feed_dict={x: vector})
        pred = sess.run(tf.nn.softmax(pred[0]))[0]
        predicts.append(pred)
        p = np.argmax(pred)

        accuracy.append(y == p)

        predi.append(p)

print()
accuracy = np.array(accuracy, dtype=np.float32)
print('Accuracy: ', np.average(accuracy))
print('F1 Score: ', f1_score(label, predi, average='macro'))
print('Precision: ', precision_score(label, predi, average='macro'))
print('Recall: ', recall_score(label, predi, average='macro'))

print(classification_report(label, predi, target_names=classes))
