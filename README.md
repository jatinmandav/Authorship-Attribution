# Authorship Attribution

## Text2Vector

Text2Vector generated word-embeddings for a given text.

#### Dependencies

* FastText - `pip3 install fasttext`
* NLTK - `pip3 install nltk`
* NumPy - `pip3 install numpy`

#### Algorithm

Word embedding model - Skipgram - trained on raw text, with target size of 100-dim.

* For every word in text
 - extract word-embedding from pre-trained model
 - Element-wise addition of word-embedding with earlier embedding
 - Averaging the word embeddings to obtain text embedding

#### Usage

Clone the repo, save word-embedding model from "Release" under the top-directory.
  ```
  from text2vector import Text2Vector

  path_to_word_embedding_model = 'skipgram-100/skipgram.bin'
  text2vec = Text2Vector(path_to_word_embedding_model)
  print('Generating embedding ..')

  text = "Any text sample here!!!"

  print(text2vec.convert(text))

  """
  [ 0.44153227  0.12361106  0.00607821  0.53515722  0.28272    -0.11348521
   -0.22341401  0.16395177 -0.01178203  0.06159557  0.01883419 -0.12652569
   -0.15079305 -0.04731999  0.20319002 -0.05538354 -0.33085034  0.37050098
    0.09624304 -0.42489241 -0.12226875 -0.2626278  -0.06713643 -0.31778053
    0.34330154  0.27109923  0.15488822  0.17079256 -0.15065777  0.1636747
   -0.12178582  0.08251881 -0.21455672  0.47482748 -0.31790714 -0.12989535
    0.22780022 -0.08566074 -0.21855019 -0.31932005 -0.13109301  0.00146381
    0.26255177  0.33713142  0.19220728 -0.10542174 -0.37171204  0.03750945
    0.1633902   0.12948658 -0.19956239 -0.1510046   0.12710889 -0.10108179
   -0.07666321  0.42590444 -0.07935622  0.07618748 -0.11696555 -0.12906784
    0.07358533 -0.34065487  0.8037034  -0.02860275  0.06394197  0.23639533
   -0.2513136  -0.28632619  0.12479368 -0.24286127  0.05824339 -0.17754047
   -0.04253732  0.34774542 -0.10136852  0.44451588  0.07106352  0.62934348
    0.38469637  0.13533326  0.00748966  0.17116917  0.14566171 -0.01778589
    0.22916658  0.22703299  0.17693119 -0.1667371  -0.44423185  0.23692855
   -0.0115481   0.081972    0.04752844  0.45813912 -0.13476718  0.23231126
    0.06251153  0.07475965  0.25681703  0.02036907]
  """

  ```
