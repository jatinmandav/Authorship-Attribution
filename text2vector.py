import fasttext
import numpy as np
from nltk.tokenize import word_tokenize

class Text2Vector:
    def __init__(self, model_path):
        print('Loading model {}'.format(model_path))
        self.model = fasttext.load_model(model_path)

    def convert(self, text):
        words = word_tokenize(text)

        embedding = []

        for word in words:
            try:
                if embedding == []:
                    embedding = np.array(self.model[word])
                else:
                    #embedding = np.linalg.multi_add([embedding, self.model[word]])
                    embedding = np.array(embedding + self.model[word])
                    embedding = embedding/2
            except Exception as e:
                pass

        return embedding

text2vector = Text2Vector('skipgram-100/skipgram.bin')
print('Generating embedding ..')
text = "Any text sample here!!!"
print(text2vector.convert(text))
