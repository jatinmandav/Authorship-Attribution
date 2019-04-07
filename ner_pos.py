import nltk
from nltk import pos_tag, RegexpParser, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.tree import Tree
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.corpus import state_union
from tqdm import tqdm
import pandas as pd

class NamedEntityRecognition:
    def __init__(self):
        self.sentTokenizer = PunktSentenceTokenizer()

    def get_pos_tag(self, posts):
        #sentences = self.sentTokenizer.tokenize(text)
        #sentences = [sent for sent in sentences if len(word_tokenize(str(sent))) > 20]

        with open('pos_tags.txt', 'w') as f:
            for i in tqdm(range(len(posts)), desc="Generating POS Tags: "):
                if len(word_tokenize(str(posts[i]))) > 10:
                    pos_tags = pos_tag(word_tokenize(str(posts[i])))
                    for tag in tqdm(pos_tags, desc='Writing Data: '):
                        f.write('{} '.format(tag[1]))

                    f.write('\n')


        return pos_tags

ner = NamedEntityRecognition()
df = pd.read_csv('data/training_blogs_data.csv', sep="|")

post = df['Post']

pos_tags = ner.get_pos_tag(post)
'''    for tag in tqdm(pos_tags, desc='Writing Data: '):
        f.write('{} '.format(tag[1]))
'''
