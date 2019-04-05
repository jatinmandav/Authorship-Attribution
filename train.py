from Models.lstm import LSTMModel
from Models.cnn import CNNModel
from Models.cnnlstm import CNNLSTMModel

from ReadData import ReadData

import os
import pandas as pd
import numpy as np
import argparse

from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Name of Model to use [lstm, cnn, cnnlstm]', required=True)
parser.add_argument('--training_csv', help='Path to Training CSV file', required=True)
parser.add_argument('--embedding', help='Path to word embedding model', default='skipgram-100/skipgram.bin')
parser.add_argument('--n_classes', help='No of classes to predict', default=6, type=int)
parser.add_argument('--optimizer', help='which Optimizer to use?', default='adam')
parser.add_argument('--batch_size', help='What should be the batch size?', default=32, type=int)
parser.add_argument('--epochs', help='How many epochs to Train?', default=100, type=int)

args = parser.parse_args()

model_list = {'lstm': LSTMModel, 'cnn': CNNModel, 'cnnlstm': CNNLSTMModel}

input_shape = (100, 1)
n_classes = args.n_classes

model = model_list[args.model](input_shape=input_shape, output_shape=n_classes)

if args.optimizer == 'adam':
    optimizer = Adam()
elif args.optimizer == 'sgd':
    optimizer = SGD()
elif args.optimizer == 'rmsprop':
    optimizer = RMSprop()
else:
    print('{} optimizer not added yet. Using Adam instead.'.format(args.optimizer))
    optimizer = Adam()


if not os.path.exists("logs_{}".format(args.model)):
	os.mkdir("logs_{}".format(args.model))

log_dir = "logs_{}/{}".format(args.model, args.optimizer)

if not os.path.exists(log_dir):
	os.mkdir(log_dir)


if not os.path.exists("weights_{}".format(args.model)):
	os.mkdir("weights_{}".format(args.model))

weights_dir = "weights_{}/{}".format(args.model, args.optimizer)

if not os.path.exists(weights_dir):
	os.mkdir(weights_dir)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

logging = TrainValTensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(weights_dir + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='loss', save_weights_only=True, save_best_only=True, period=3)
earlystopper = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1, mode='auto')

reader = ReadData('data/training_blogs_data.csv', 'embeddings/skipgram-100/skipgram.bin', batch_size=args.batch_size)
generator = reader.read()

model.fit_generator(generator=generator, steps_per_epoch=int(reader.data_size/args.batch_size),
          epochs=args.epochs, verbose=1,
          callbacks=[tensorboard, reduce_lr, earlystopper, checkpoint])

m.save_weights(os.path.join(weights_dir, "final_weights.model"))
m.save(os.path.join(weights_dir, "final.model"))
