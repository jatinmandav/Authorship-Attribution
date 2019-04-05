import fasttext
import numpy as np
from nltk.tokenize import word_tokenize

class Text2Vector:
    def __init__(self, model_path):
        print('Loading embedding model {}'.format(model_path))
        self.model = fasttext.load_model(model_path)

    def convert(self, text):
        #words = word_tokenize(text)
        words = text.split(' ')

        vector = []

        for word in words:
            try:
                if vector == []:
                    vector = np.array(self.model[word])
                else:
                    #embedding = np.linalg.multi_add([embedding, self.model[word]])
                    vector = np.array(vector + self.model[word])
                    vector = vector/2
            except Exception as e:
                print('In text2vector.py: {}'.format(e))

        vector = np.reshape(vector, (vector.shape[0], 1))
        return vector


if __name__ == "__main__":
    text2vector = Text2Vector('embeddings/skipgram-100/skipgram.bin')
    print('Generating embedding ..')
    text = "Any text sample here!!!"
    vector = text2vector.convert(text)
    print(vector.shape)
