from text2vector import Text2Vector

import os
import pandas as pd
import numpy as np

class ReadData:
    def __init__(self, path_csv, embedding_model, batch_size=32, train_val_split=0.1):
        self.text2vec = Text2Vector(embedding_model)
        self.data = pd.read_csv(path_csv, sep="|")
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data_size = len(self.data)

        self.train = self.data.head(int(self.data_size*(1-train_val_split))).reset_index(drop=True)
        self.train_size = len(self.train)
        self.val = self.data.tail(int(self.data_size*train_val_split)).reset_index(drop=True)
        self.val_size = len(self.val)

        self.classes = ['male10', 'male20', 'male30', 'female10', 'female20', 'female30']
        self.batch_size = batch_size

    def get_embedding(self, text):
        return self.text2vec.convert(text)

    def generate_val_batch(self):
        no_batches = int(len(self.val['Post'])/self.batch_size)
        while True:
            start_index = 0
            for i in range(no_batches-1):
                vectors = []
                labels = []
                j = start_index
                while (start_index <= j < start_index + self.batch_size):
                    if len(str(self.val['Post'][j])) < 2:
                        j += 1
                        continue
                    try:
                        label = '{}{}'.format(self.val['Gender'][j], self.val['Age_Group'][j])
                        one_hot = np.zeros(len(self.classes))
                        one_hot[self.classes.index(label)] = 1
                        labels.append(one_hot)

                        vector = self.get_embedding(str(self.val['Post'][j]))
                        vectors.append(vector)

                        j += 1
                    except Exception as e:
                        print(e)

                start_index += self.batch_size

                vectors = np.array(vectors)
                labels = np.array(labels)

                yield vectors, labels

    def generate_train_batch(self):
        no_batches = int(len(self.train['Post'])/self.batch_size)
        while True:
            start_index = 0
            for i in range(no_batches-1):
                vectors = []
                labels = []
                j = start_index
                while (start_index <= j < start_index + self.batch_size):
                    if len(str(self.train['Post'][j])) < 2:
                        j += 1
                        continue
                    try:
                        label = '{}{}'.format(self.train['Gender'][j], self.train['Age_Group'][j])
                        one_hot = np.zeros(len(self.classes))
                        one_hot[self.classes.index(label)] = 1
                        labels.append(one_hot)

                        vector = self.get_embedding(str(self.train['Post'][j]))
                        vectors.append(vector)

                        j += 1
                    except Exception as e:
                        print(e)

                start_index += self.batch_size

                vectors = np.array(vectors)
                labels = np.array(labels)

                yield vectors, labels

if __name__ == "__main__":
    reader = ReadData('data/training_blogs_data.csv', 'embeddings/skipgram-100/skipgram.bin')
    for v, l in reader.generate_val_batch():
        print(v.shape, l.shape)
