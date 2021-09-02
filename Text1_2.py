#문3
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
import re

#연설문 텍스트파일은 enter가 삽입되지 안은 한 줄짜리 데이터
f = open("Data/연설문.txt")
lines = f.readlines()[0]
f.close()
print(lines)
print()

tokenizer = RegexpTokenizer('[\w]+')    # 토큰을 만들 때 제거할 옵션 설정...
stop_words = stopwords.words('english')#시험문제 : 'english's
#전처리 과정
words = lines.lower()   #시험문제
tokens = tokenizer.tokenize(words)
print()
print(tokens)
print(len(tokens))
print()

# 불용어 제거 -1차
middle_tokens = [i for i in tokens if i not in stop_words] #불용어 제거
print(len(middle_tokens))

last_tokens = [i for i in middle_tokens if len(i) > 1]  #시험문제 : 한글자 제거. len(i) > 1
print()
print(last_tokens)
print(len(last_tokens))

#중단확인단계
result = pd.Series(last_tokens).value_counts().head(30) #시험문제
print(result)

# 2차 불용어 처리
sword = ['never','many','today','back','make','across','president']
result_tokens = [i for i in last_tokens
                 if i not in sword]
print(result_tokens)
print()

# 어근의 동일화
ps_stemmer = PorterStemmer()
stem = []

for w in result_tokens:
    stem.append(ps_stemmer.stem(w)) #시험문제 : stem(w)
summary = []
for a in stem:
    if a == 'american':
        summary.append(a.replace('american','america'))
    elif a == 'countri':
        summary.append(a.replace('countri','country'))
    elif a == 'peopl':
        summary.append(a.replace('peopl','people'))
    else:
        summary.append(a)
# 단어별 빈도수 집계하기
data = Counter(summary)
data2 = data.most_common(15)
print(data2)

for i in data2:
    print(i[0],i[1])
print()

words = dict(data2) #시험 : 전체
print(words)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(font_path='C:/Windows/Fonts/HMFMPYUN.TTF',    #영어폰트명 가져오기
                      relative_scaling=0.4, background_color='white').generate_from_frequencies(words)  #시험
plt.figure(figsize=(8,4))
plt.imshow(wordcloud)
plt.title('Speech',fontsize=30)
plt.axis('off')
plt.show()