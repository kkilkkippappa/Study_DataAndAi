import pandas as pd
from konlpy.tag import Hannanum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
hannanum = Hannanum()
Data = pd.read_csv('Data/군집분석.csv') #시험 나옴
docs = []
for i in Data(['기사내용']):
    docs.append(hannanum.nouns(i))
print()

for i in range(len(docs)):
    docs[i] = ''.join(docs[i])
vec = CountVectorizer()
X = vec.fit_transform(docs)#시험문제 docs
print(vec.get_feature_names())#시험문제
print()

#군집분석할 데티터(최종)
df = pd.DataFrame(X.torray(), columns=vec.get_feature_names()) #시험문제 : X.toarray()
print(df)

print('분할군집분석 : kmenas 군집분석')#시험문제
kmeans = KMeans(n_clusters=3).fit(df)#시험문제 fit(df)
print(kmeans.labels_)#시험문제 : kmeans.labels_