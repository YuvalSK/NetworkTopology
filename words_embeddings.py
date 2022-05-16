# -*- coding: utf-8 -*-
"""
using pretrained language models to extract context based  on:
1) text-level similarity
"""
import warnings
warnings.filterwarnings('ignore')
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import pandas as pd
from math import sqrt
import seaborn as sns
from sklearn.metrics import pairwise
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def twitter(pos_words, n_words, neg_words = []):
    print('Loading pretrained model...')
    model_glove_twitter = api.load("glove-twitter-25") # twitter dataset
    
    if neg_words == []:
        emb_words = model_glove_twitter.most_similar(positive = pos_words, topn = n_words)
    else:   
        emb_words = model_glove_twitter.most_similar(positive = pos_words, negative=neg_words, topn = n_words)
    #print(emb_words)
    return emb_words

'''
def wiki(words, n_words):
    print('Loading...')
    model_gigaword = api.load("glove-wiki-gigaword-100") # wikipedia pretrained model
    emb_words = model_gigaword.most_similar(positive = words,topn = n_words)
    #print(emb_words)
    return emb_words
'''

total_words = 1000000
'''
#Personality 
## https://gosling.psy.utexas.edu/scales-weve-developed/ten-item-personality-measure-tipi/ten-item-personality-inventory-tipi/

# from TIPI, we get the words for extravesrion:
pos_words = ['outgoing','enthusiastic'] # high on extraversion 
sim_words = twitter(pos_words, n_words = total_words)
sim_words = dict(sim_words)

neg_words = ['reserved','quiet'] # low on extraversion
disim_words = twitter(neg_words, n_words = total_words)    
disim_words = dict(disim_words)

# from TIPI, we get the words for neuroticism:
pos_words = ['anxious','upset'] # high on N 
sim_words = twitter(pos_words, n_words = total_words)
sim_words = dict(sim_words)

neg_words = ['calm','stable'] # low on N
disim_words = twitter(neg_words, n_words = total_words)    
disim_words = dict(disim_words)

'''
# Depression
## https://ieeexplore.ieee.org/document/6784326

# from LJ, the words are:
pos_words = ['sadness','anxious'] # high on depression 
sim_words = twitter(pos_words, n_words = total_words)
sim_words = dict(sim_words)

neg_words = ['ingestion','home','sexual'] # low on depression
disim_words = twitter(neg_words, n_words = total_words)    
disim_words = dict(disim_words)
 

#dois = ['acting', 'animals', 'anime', 'art', 'basketball', 'biking', 'books', 'boys', 'camping', 'candles', 'cars', 'cats', 'cheese', 'chocolate', 'clothes', 'coffee', 'computers', 'concerts', 'cooking', 'dancing', 'dogs', 'drawing', 'dreams', 'drinking', 'driving', 'dvds', 'eating', 'family', 'family guy', 'fanfiction', 'fantasy', 'fashion', 'food', 'football', 'friends', 'girls', 'green day', 'guitar', 'guys', 'harry potter', 'hiking', 'history', 'hugs', 'internet', 'johnny depp', 'kissing', 'laughing', 'life', 'linkin park', 'literature', 'lord of the rings', 'love', 'manga', 'movies', 'music', 'nirvana', 'painting', 'philosophy', 'Photography', 'pictures', 'piercings', 'poetry', 'politics', 'psychology', 'punk', 'radiohead', 'rain', 'reading', 'rock', 'running', 'sex', 'shoes', 'shopping', 'singing', 'sleep', 'sleeping', 'snow', 'snowboarding', 'soccer', 'stars', 'summer', 'swimming', 'taking back sunday', 'talking', 'tattoos', 'the beatles', 'the used', 'travel', 'traveling', 'tv', 'vampires', 'video games', 'women', 'writing']

# only most popular
#dois = ['music','movies', 'reading','writing','friends','art','photography','books']
'''
full list included 10 words with 2 parts so I omitted them:
    - family guy
    - green day
    - harry potter
    - johnny depp
    - linkin park
    - lord of the rings
    - taking back sunday
    - the beatles
    - the used
    - video games
'''
dois = ['acting', 'animals', 'anime', 'art', 'basketball', 'biking', 'books', 'boys', 'camping', 'candles',
        'cars', 'cats', 'cheese', 'chocolate', 'clothes', 'coffee', 'computers', 'concerts', 'cooking', 'dancing',
        'dogs', 'drawing', 'dreams', 'drinking', 'driving', 'dvds', 'eating', 'family', 'fanfiction', 'fantasy',
        'fashion', 'food', 'football', 'friends', 'girls', 'guitar', 'guys', 'hiking', 'history', 'hugs', 'internet', 
        'kissing', 'laughing', 'life', 'literature', 'love', 'manga', 'movies', 'music', 'nirvana', 'painting', 'philosophy',
        'photography', 'pictures', 'piercings', 'poetry', 'politics', 'psychology', 'punk', 'radiohead', 'rain', 'reading', 
        'rock', 'running', 'sex', 'shoes', 'shopping', 'singing', 'sleep', 'sleeping', 'snow', 'snowboarding', 'soccer', 'stars',
        'summer', 'swimming', 'talking', 'tattoos', 'travel', 'traveling', 'tv', 'vampires', 'women', 'writing']

pos_scores = []
neg_scores = []
for doi in dois:
    print(doi)
    doi_pos_score = 0
    doi_neg_score = 0
    count = 0
    if doi in sim_words.keys():
        doi_pos_score = sim_words[doi]
        print(f'-----found in pos, score: {doi_pos_score}')
        count += 1
    else:
        break
    if doi in disim_words.keys():
        doi_neg_score = disim_words[doi]
        print(f'-----found in neg, score: {doi_neg_score}')
        count += 1
    else:
        break
    pos_scores.append(doi_pos_score)
    neg_scores.append(doi_neg_score)
    
d = {'doi': dois, 'positive_score': pos_scores, 'negative_scores': neg_scores}
df = pd.DataFrame(d)
df.to_csv('Data/dictionaries/NLP/doi2dep.csv')

#######################################################################
        
scores = []
for doi in dois:
    print(doi)
    doi_pos_score = 0
    doi_neg_score = 0
    count = 0
    if doi in sim_words.keys():
        doi_pos_score = sim_words[doi]
        print(f'-----found in pos, score: {doi_pos_score}')
        count += 1
    if doi in disim_words.keys():
        doi_neg_score = disim_words[doi]
        print(f'-----found in neg, score: {doi_neg_score}')
        count += 1
        
    t_score = doi_pos_score + doi_neg_score
    scores.append(t_score/count)
    
    

'''
2) sentence level similarity ???
'''
# https://arxiv.org/pdf/1803.11175.pdf
def nnlm():
    embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-de-dim50/1")
    return embed

embed = nnlm()
tem = "I see myself as calm and emotionally stable"
x = tf.constant([tem])
embeddings = embed(x)
x = np.asarray(embeddings)
x1 = x[0].tolist()


tem = "I am anxious and easily upset"
x = tf.constant([tem])
embeddings = embed(x)
x = np.asarray(embeddings)
x2 = x[0].tolist()

def cosineSim(a1,a2):
    sum = 0
    suma1 = 0
    sumb1 = 0
    for i,j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        sum += i*j
    cosine_sim = sum / ((sqrt(suma1))*(sqrt(sumb1)))
    return cosine_sim

print(cosineSim(x1,x2))


'''
pos_twitter_dict = model_glove_twitter.most_similar(positive=['depression'],topn=1000)
pos_wiki_dict = model_gigaword.most_similar(positive=['depression'],topn=1000)

df = pd.DataFrame(pos_twitter_dict)
df.to_csv('pos_dep_twitter.csv')

df2 = pd.DataFrame(pos_wiki_dict)
df2.to_csv('pos_dep_wiki.csv')
'''

