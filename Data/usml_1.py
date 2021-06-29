# import mglearn
# import matplotlib.pyplot as plt
#
# #kmeans-평균 군집 알고리즘의 수행과정
# mglearn.plots.plot_kmeans_algorithm()
# plt.show()
#
# #[k-평균 알고리즘으로 찾은 클러스터 중심과 클러스터 경계]
# mglearn.plots.plot_kmeans_boundaries()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans
#
# X = np.array([
#     [6,3],[11,15],[17,12],[24,10],[20,25],[22,30],
#     [85,70],[71,81],[60,79],[56,52],[81,91],[80,81]
# ])
# #입력한 데이터들을 그래프로 표현
# plt.scatter(X[:,0],X[:,1],edgecolors='orange', color='limegreen')
# plt.title('original_data', fontsize=15)
# plt.show()
#
# #kmeans객체를 만들고 매개변수 n_clusters의 값으로 2를 전달
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X)
#
# #알고리즘이 생성한 클러스터의 중심값
# print('알고리즘이 생성한 클러스터의 중심점의 좌표')
# print(kmeans.cluster_centers_)
#
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:,0], centers[:,1],c='green',s=200,alpha=0.5)
#
# print('0과 1은 단지 클러스터 번호')
# print(kmeans.labels_)
#
# plt.scatter(X[:,0],X[:,1], c=kmeans.labels_,cmap='rainbow')
#
# for i in range(12):
#     plt.text(X[i,0],X[i,1],kmeans.labels_[i],c='green')
# plt.title('cluster_centers and labels', fontsize=15)
# plt.show()

# from sklearn.datasets import load_iris
# iris = load_iris()
# print('iris의 keys\n{}'.format(iris.keys()))
# print()
#
# print('iris의 data의 크기 : \n{}'.format(iris['data'].shape))
# print(iris.data[:3,:])
# print()
# print('features_names')
# print(iris.feature_names)
#
# #Kmeans start
# from sklearn.cluster import KMeans
# X = iris.data[:,:2]
# y = iris.target
#
# #raw data 확인
# plt.scatter(X[:,0],X[:,1],c=y,cmap='gist_rainbow')
# plt.xlabel('Speal Length', fontsize=10)
# plt.ylabel('Sepal Width', fontsize=10)
# plt.show()
#
# #K-mean fit
# km = KMeans(n_clusters=3)
# km.fit(X)
#
# #학습의 결과
# print(km.labels_)
# #중심점 위치
# centers= km.cluster_centers_
# print(centers)
#
# #시각화
# new_labels = km.labels_
# fig, axes = plt.subplots(1,2,figsize=(10,5))
# axes[0].scatter(X[:,0],X[:,1],c=y,cmap='gist_rainbow',edgecolor='k', s=50) #c 빈칸
# axes[0].scatter(X[:,0],X[:,1],c=new_labels,cmap='gist_rainbow',edgecolor='k', s=50)#c부분 빈칸
#
# axes[0].set_xlabel('Sepal length', fontsize=10)
# axes[0].set_ylabel('Sepal width', fontsize=10)
# axes[1].set_xlabel('Sepal length', fontsize=10)
# axes[1].set_xlabel('Sepal width', fontsize=10)
#
# axes[0].set_title('Actual_labels', fontsize=15)
# axes[1].set_title('Predicted_labels', fontsize=15)
# plt.show()

# import numpy as np
# #논리적 인덱싱
# print('논리적 인덱싱')
# ages = np.array([18,19,20,21,25])
# print(ages>=20)
# print(ages[ages>=20])
# print()
#
# #부울리언 인덱싱
# names=np.array(['a','b','c','d','e','f','g'])
# data=np.random.randint(1,100,size=(7,4))
# print(data)
# print(names=='a')
# print()
# print(data[names=='a'])
# print()
# print((names=='b') | (names=='g'))
# print()
# print(data[(names=='b') | (names=='g')])
# print()
#
# problems = np.array([[18,30],[19,39],[20,30],[21,41],[15,27]])
# answers = np.array([0,0,1,1,0])
# print(problems)
# print()
#
# print(answers==0)
# print()
# print(answers[answers==0])
# print()
# print(answers==1)
# print(problems[answers==1])
# print()
# print(problems[answers==0,0])
# print(problems[answers==0,1])
#
# #한글설정
# import matplotlib
# import matplotlib.font_manager as fm
# fm.get_fontconfig_fonts()
#
# font_location = 'C:/Windows/Fonts/HMFMPYUN.TTF'
# font_name = fm.FontProperties(fname=font_location).get_name()
# matplotlib.rc('font',family=font_name)
#
# d_length = [77,78,85,83,73,77,73,80,75,77,86,86,79,83,83,88]
# d_height = [25,28,19,30,21,22,17,35,56,57,50,53,60,53,49,61]
#
# m_length = [34,38,38,41,30,37,41,35,65,67,76,76,69,63,63,78]
# m_height = [22,25,19,30,21,24,28,18,50,50,50,48,50,50,45,51]
#
# d= zip(d_length,d_height)#시험문제
# l = list(d)
# X1 = [list(i) for i in l]#시험문제
#
# m = zip(m_length, m_height)
# l=list(m)
# X2 = [list(i) for i in l]
#
# #입력데이터셋
# dogs = X1+X2
# X = np.array(dogs)#시험문제
#
# plt.scatter(X[:,0], X[:,1],c='orange',marker='o',edgecolors='black',s=50)
# plt.title('before clustering : 강아지 크기와 길이')
# plt.xlabel('길이')
# plt.ylabel('높이')
# plt.show()
#
# #clustering
# km = KMeans(n_clusters=3)
# km.fit(X)
# y_km = km.labels_
#
# plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='s', edgecolors='black',label='cluster 1')#시험문제
# plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='s', edgecolors='black',label='cluster 2')#시험문제
# plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='blue', marker='s', edgecolors='black',label='cluster 2')#시험문제
#
# #plot the centroids
# print(print(km.cluster_centers_))
# plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],s=250,marker='*',c='red',edgecolors='black',label='centroids')
# plt.legend(scatterpoints=1)
# plt.grid()
#
# plt.xlabel('Length')
# plt.ylabel('Height')
# plt.title('강아지 크기와 길이')
# plt.legend(loc='upper left')
# plt.show()

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#step1. 데이터 적재하기와 살펴보기
iris = load_iris()

from sklearn.cluster import KMeans
print(iris.feature_names)
print()
X=iris.data[:,2:]
y=iris.target

km = KMeans(n_clusters=3)
km.fit(X)
centers = km.cluster_centers_

new_labels = km.labels_
fig, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].scatter(X[:,0],X[:,1],c=y,cmap='gist_rainbow',edgecolor='k', s=50) #c 빈칸
axes[0].scatter(X[:,0],X[:,1],c=new_labels,cmap='gist_rainbow',edgecolor='k', s=50)#c부분 빈칸

axes[0].set_xlabel('petal length', fontsize=10)
axes[0].set_ylabel('petal width', fontsize=10)
axes[1].set_xlabel('petal length', fontsize=10)
axes[1].set_xlabel('petal width', fontsize=10)

X1 = np.array([[1.4,0.2],[4.7,1.4],[6.0,2.5],[4.0,1.7],[3.0,1.2]])
print(X1)

y_pred = km.predict(X1)
plt.scatter(X[:,0],X[:,1],c=y,cmap='gist_rainbow',edgecolor='k', s=50)
plt.scatter(X1[:,0],X1[:,1],c=y_pred,cmap='rainbow',edgecolor='k',marker='*',s=300)
plt.axis([0,8,-0.5,3.5])
plt.title('random clustering test2')
plt.show()

X = np.array([
    [6,3],[11,15],[17,12],[24,10],[20,25],[22,30],[85,70],[71,81],[60,79],[56,52],[81,91],[80,81]
])
kmeans = KMeans(n_clusters=2).fit(X)
y = kmeans.labels_
centers = kmeans.cluster_centers_

new = np.array([[10,10],[60,65]])
y_pred = kmeans.predict(new)
new_data = ['(10,10)','(60,50)']

#시각화
plt.scatter(X[:,0],X[:,1], c=y,edgecolors='orange', cmap='rainbow',s=50)
plt.scatter(centers[:,0],centers[:,1], c='yellowgreen',s=200, alpha=0.5)
data='중심점'
for i in range(2):
    plt.text(centers[i,0],centers[i,1],data, color='green')#시험문제
plt.scatter(new[:,0],new[:,1],c=y_pred,s=300, alpha=0.5, cmap='gist_rainbow')
for i in range(2):
    plt.text(new[i,0],new[i,1],new_data[i],color='green')#시험문제
plt.title('클리스터링, k=2', fontsize=20)
plt.show()