from konlpy.tag import Hannanum
from konlpy.tag import Kkma
from konlpy.tag import *

import re
from konlpy.tag import Okt
import pandas as pd
okt = Okt()

original_data = '물가상승률에 대하여 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.'
token=re.sub('(\.)','',original_data)   # 온점만 없앰.
print(token)
print()

token = okt.morphs(token)   #시험문제
print(token)
print()

print('불용어 제거 1')
data_set = []
for i in token:
    if len(i) > 1:  #시험문제
        data_set.append(i)
print(data_set)
print()

print('불용어 제거2')
sword = ['하는','대충'] #시험문제
token = [i for i in data_set
         if i not in sword]
print(token)
print()

word_dict = {}
words_cnt_list={}

for word in token:
    if word not in word_dict:
        word_dict[word] = len(word_dict)   #시험문제 # 각 단어에 고유한 정수 인덱스 부여   #len(word_dict) 중요
print()
print(word_dict)
print()

words_cnts = []
for word in word_dict.keys():
    words_cnts.append(original_data.count(word))    #original_data.count(word) 부분 중요! #시험문제
print(words_cnts)

word_list = []  #시험문제
for word in word_dict.keys():   #시험문제
    word_list.append(word)  #시험문제
print(word_list)    #시험문제

print()
df = pd.DataFrame([words_cnts], columns=word_list)  #시험문제
print(df)

# 다음 주 나올 시험문제
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
ps_stemmer = PorterStemmer()
corpus = [
    'love changes everything',
    'you know I want your love',
    'I do what you want',
    'you can do as you want',
    'for your future'
]
sw = ['as','what','can','for','your','you'] #불용어    #시험문제
vector = CountVectorizer(stop_words=sw) # 시험문제 : stop_words=sw
tf = vector.fit_transform(corpus).toarray() # 시험문제

stem = []
result = vector.get_feature_names()# 시험문제
for w in result:
    stem.append(ps_stemmer.stem(w)) # 시험문제 : stem(w)
print(stem)
print(tf)

# 8dnjf