from text2vector import Text2Vector

import os
import pandas as pd
import numpy as np

class ReadData:
    def __init__(self, path_csv, embedding_model, batch_size=32):
        self.text2vec = Text2Vector(embedding_model)
        self.data = pd.read_csv(path_csv, sep="|")
        self.data_size = len(self.data['Post'])
        self.classes = ['male10', 'male20', 'male30', 'female10', 'female20', 'female30']
        self.batch_size = batch_size

    def get_embedding(self, text):
        return self.text2vec.convert(text)

    def read(self):
        no_batches = int(len(self.data['Post'])/self.batch_size)
        while True:
            start_index = 0
            for i in range(no_batches-1):
                vectors = []
                labels = []
                j = start_index
                while (start_index <= j < start_index + self.batch_size):
                    if len(str(self.data['Post'][j])) < 2:
                        j += 1
                        continue
                    try:
                        label = '{}{}'.format(self.data['Gender'][j], self.data['Age_Group'][j])
                        one_hot = np.zeros(len(self.classes))
                        one_hot[self.classes.index(label)] = 1
                        labels.append(one_hot)

                        vector = self.get_embedding(str(self.data['Post'][j]))
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
    for v, l in reader.read():
        print(v.shape, l.shape)
