# from gensim.models import FastText
#
# model = FastText.load_fasttext_format('../../../datasets/crawl-300d-2M.vec')
# vector = model['apple']
# print(vector)
import time

from gensim.models import keyedvectors

model = keyedvectors.load_word2vec_format('../../../datasets/crawl-300d-2M.vec')
vector = model['apple']
print(type(vector))
