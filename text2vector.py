import fasttext
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk.data
from collections import Counter
import pandas as pd

class Text2Vector:
    def __init__(self, embed_path, size):
        print('Loading embedding model {}'.format(embed_path))
        self.embed_model = fasttext.load_model(embed_path)
        
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.size = size

    def calculate_tf_idf(self, raw_count, max_raw_count_in_document, no_documents, no_documents_in_which_word_occured):
        tf = 0.5 + 0.5*(raw_count/max_raw_count_in_document)
        idf = np.log(no_documents/(1 + no_documents_in_which_word_occured))
        return tf*idf

    def convert(self, text):
        raw_words = word_tokenize(text)
        #words = text.split(' ')
        sentences = self.sent_detector.tokenize(text)
        for i in range(len(sentences)):
            sentences[i] = word_tokenize(sentences[i])

        vocab_words = set(raw_words)

        word_count = Counter(raw_words)
        raw_count = Counter(raw_words)
        max_raw_count_in_sent = next(iter(word_count.values()))

        for word in word_count:
            count = 0
            for sent in sentences:
                if word in sent:
                    count += 1

            no_sent_in_which_word_occured = count
            word_count[word] = self.calculate_tf_idf(word_count[word], max_raw_count_in_sent, len(sentences), no_sent_in_which_word_occured)

        data_dict = {'word': list(word_count.keys()), 'TF_IDF': list(word_count.values())}
        df = pd.DataFrame.from_dict(data_dict).reset_index(drop=True)
        df = df.sort_values('TF_IDF', ascending=True).reset_index(drop=True)
        df = df.head(self.size[0])

        vectors = []

        for i, word in enumerate(df['word']):
            try:
                embed_vector = np.array(self.embed_model[word])
                vectors.append(list(embed_vector) + [df['TF_IDF'][i]])
            except Exception as e:
                print('In text2vector.py: {}'.format(e))

        for i in range(self.size[0] - len(vectors)):
            vectors.append(np.zeros(self.size[1]))

        vectors = np.array(vectors)
        return vectors


if __name__ == "__main__":
    text2vector = Text2Vector('embeddings/skipgram-100/skipgram.bin', 'embeddings/skipgram-pos-100/skipgram_pos.bin', (100, 201))
    print('Generating embedding ..')
    text = "Any text here!!!!!"
    vector = text2vector.convert(text)
    print(vector)
    print(vector.shape)
