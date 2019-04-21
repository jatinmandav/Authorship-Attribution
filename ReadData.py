from text2vector import Text2Vector

import os
import pandas as pd
import numpy as np

class ReadData:
    def __init__(self, path_csv, embedding_model, classes, batch_size=32, no_samples=10000, train_val_split=0.1):
        self.text2vec = Text2Vector(embedding_model, size=(75, 101))
        self.data = pd.read_csv(path_csv, sep="|")
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True).head(no_samples)
        self.data_size = len(self.data)

        self.train = self.data.head(int(self.data_size*(1-train_val_split))).reset_index(drop=True)
        self.train_size = len(self.train)
        self.val = self.data.tail(int(self.data_size*train_val_split)).reset_index(drop=True)
        self.val_size = len(self.val)

        self.classes_category = classes
        self.classes = self.get_classes()

        self.batch_size = batch_size

    def get_classes(self):
        return [class_ for class_ in open(self.classes_category + '.txt', 'r').read().split('\n') if len(class_) > 1]

    def get_embedding(self, text):
        return self.text2vec.convert(text)

    def get_next_batch(self, start, end):
        vectors = []
        labels = []
        for i in range(start, end):
            try:
                if len(str(self.train['Post'][i]).split()) < 15:
                    continue

                #label = '{}{}'.format(self.val[self.classes_category][j], self.val['Age_Group'][j])
                label = '{}'.format(self.train[self.classes_category][i])
                one_hot = np.zeros(len(self.classes))
                one_hot[self.classes.index(label)] = 1
                labels.append(one_hot)

                vector = self.get_embedding(str(self.train['Post'][i]))
                vectors.append(np.array(vector))
            except Exception as e:
                print(e, self.train['Post'][i])

        vectors = np.array(vectors)
        labels = np.array(labels)

        return vectors, labels

    def read_all_train(self):
        vectors = []
        labels = []

        for i in range(self.train_size):
            try:
                if len(str(self.train['Post'][i]).split()) < 15:
                    continue

                #label = '{}{}'.format(self.val[self.classes_category][j], self.val['Age_Group'][j])
                label = '{}'.format(self.train[self.classes_category][i])
                one_hot = np.zeros(len(self.classes))
                one_hot[self.classes.index(label)] = 1
                labels.append(one_hot)

                vector = self.get_embedding(str(self.train['Post'][i]))
                vectors.append(np.array(vector))
            except Exception as e:
                print(e, self.train['Post'][i])

        vectors, labels = np.array(vectors), np.array(labels)
        return vectors, labels

    def read_all_val(self):
        vectors = []
        labels = []

        for i in range(self.val_size):
            try:
                if len(str(self.val['Post'][i]).split()) < 15:
                    continue
                #label = '{}{}'.format(self.val[self.classes_category][j], self.val['Age_Group'][j])
                label = '{}'.format(self.val[self.classes_category][i])
                one_hot = np.zeros(len(self.classes))
                one_hot[self.classes.index(label)] = 1
                labels.append(one_hot)

                vector = self.get_embedding(str(self.val['Post'][i]))
                vectors.append(vector)
            except Exception as e:
                print('ReadData: ', e)

        return np.array(vectors), np.array(labels)

    def generate_val_batch(self):
        no_batches = int(len(self.val['Post'])/self.batch_size)
        while True:
            start_index = 0
            for i in range(no_batches):
                vectors = []
                labels = []
                j = start_index
                while (start_index <= j < start_index + self.batch_size):
                    if len(str(self.val['Post'][j])) < 2:
                        j += 1
                        continue
                    try:
                        #label = '{}{}'.format(self.val[self.classes_category][j], self.val['Age_Group'][j])
                        label = '{}'.format(self.val[self.classes_category][j])
                        one_hot = np.zeros(len(self.classes))
                        one_hot[self.classes.index(label)] = 1
                        labels.append(one_hot)

                        vector = self.get_embedding(str(self.val['Post'][j]))
                        vectors.append(vector)

                        j += 1
                    except Exception as e:
                        print('ReadData: ', e)

                start_index += self.batch_size

                vectors = np.array(vectors)
                labels = np.array(labels)

                yield vectors, labels

    def generate_train_batch(self):
        no_batches = int(len(self.train['Post'])/self.batch_size)
        while True:
            start_index = 0
            for i in range(no_batches):
                vectors = []
                labels = []
                j = start_index
                while (start_index <= j < start_index + self.batch_size):
                    if len(str(self.train['Post'][j])) < 2:
                        j += 1
                        continue
                    try:
                        #label = '{}{}'.format(self.train[self.classes_category][j], self.train['Age_Group'][j])
                        label = '{}'.format(self.train[self.classes_category][j])
                        one_hot = np.zeros(len(self.classes))
                        one_hot[self.classes.index(label)] = 1
                        labels.append(one_hot)

                        vector = self.get_embedding(str(self.train['Post'][j]))
                        vectors.append(vector)

                        j += 1
                    except Exception as e:
                        print('ReadData: ', e)

                start_index += self.batch_size

                vectors = np.array(vectors)
                labels = np.array(labels)

                yield vectors, labels

if __name__ == "__main__":
    reader = ReadData('data/training_blogs_data.csv', 'embeddings/skipgram-100/skipgram.bin', 'embeddings/skipgram-pos-100/skipgram_pos.bin')
    #for v, l in reader.generate_val_batch():
    #    print(v.shape, l.shape)

    generator = reader.generate_val_batch
    x, y = generator()
    x2, y2 = generator()
    print(x == x2, y == y2)
