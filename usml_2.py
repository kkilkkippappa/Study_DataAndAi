#unsupervised Learning

#한글설정
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()

font_location = 'C:/Windows/Fonts/HMFMPYUN.TTF'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font',family=font_name)

# #데이터 세트
# X = np.array([
#     [6,3],[11,15],[17,12],[24,10],[20,25],[22,30],[85,70],[71,81],[60,79],[56,52],[81,91],[80,81]
# ])
#
# # kmeans 객체를 만들고 매개변수 n_clusters의 값으로 2를 전달.
# kmeans = KMeans(n_clusters=2).fit(X)#시험문제
# y = kmeans.labels_ #시험문제. kmeans의 정답.
# centers = kmeans.cluster_centers_ #시험문제, 중심점.
#
# # new data
# new = np.array([[10,10],[60,65]])# np.array(2차원 배열)
# y_pred = kmeans.predict(new) #시험문제
# new_data = ['(10,10)','(60,50)']
#
# #시각화
# #시각화
# plt.scatter(X[:,0],X[:,1], c=y,edgecolors='orange', cmap='rainbow',s=50)#시험문제
# plt.scatter(centers[:,0],centers[:,1], c='yellowgreen',s=200, alpha=0.5)#시험문제
# data='중심점'
# for i in range(2):
#     plt.text(centers[i,0],centers[i,1],data, color='green')#시험문제
# plt.scatter(new[:,0],new[:,1],c=y_pred,s=300, alpha=0.5, cmap='gist_rainbow')
# for i in range(2):
#     plt.text(new[i,0],new[i,1],new_data[i],color='green')#시험문제
# plt.title('클리스터링, k=2', fontsize=20)
# plt.show()

#문제 2. iris 데이터
from sklearn.datasets import load_iris
iris = load_iris()
#print(iris_df.head())

#아이리스의 문제와 정답
X = iris.data #시험문제
y = iris.target #시험문제

#학습 데이터와 테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,random_state=4) #시험문제
print('X_train 의 크기 : {}'.format(X_train.shape))
print('y_train 의 크기 : {}'.format(y_train.shape))

#학습데이터를 이용하여 kmeans 알고리즘 수행
X = X_train[:,:2] #시험문제
y_sepal = y_train #시험문제

km = KMeans(n_clusters=3) #,시험문제
km.fit(X) #시험문제
#학습의 결과
labels = km.labels_ #시험문제
#중심점 위치
centers = km.cluster_centers_ #시험문제

#시각화
fig, axes = plt.subplots(1,2,figsize=(10,5))
axes[0].scatter(X[:,0],X[:,1],c=y_sepal,cmap='gist_rainbow',edgecolor='k', s=50) #c 빈칸
axes[1].scatter(X[:,0],X[:,1],c=labels,cmap='rainbow',edgecolor='k', s=50)#c부분 빈칸

axes[0].set_xlabel('꽃받침 길이', fontsize=10, c='orange')
axes[0].set_ylabel('꽃받침 너비', fontsize=10)
axes[1].set_xlabel('꽃받침 길이', fontsize=10, c='orange')
axes[1].set_ylabel('꽃받침 너비', fontsize=10)
axes[0].set_title('붓꽆의 기본 클러스터링', fontsize=15, c='purple')
axes[1].set_title('꽃받침의 길이와 너비의 클러스터링(KMeans)', fontsize=15, c='purple')
axes[1].scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
                s=250, marker='*', c='yellow', edgecolor='black', label = 'centroids')
plt.show()

#테스트 데이터
X1 = np.array([[5.4,5.2],[5.7,3.1],[6.0,2.4],[6.3,8.4],[5.1,3.0]])    #시험문제
y_pred = km.predict(X1) #시험문제

plt.scatter(X[:,0], X[:,1], c=labels, cmap='gist_rainbow',edgecolors='k',s=50)
plt.scatter(X1[:,0], X1[:,1], c=y_pred, cmap='rainbow',edgecolors='k',marker='*', s=300)
plt.xlabel('Length', fontsize=10, c='orange')
plt.ylabel('Height', fontsize=10, c='green')
plt.title('테스트 데이터', fontsize=15, c='purple')
plt.show()

# #문제3. dog
# d_length = [77,78,85,83,73,77,73,80,75,77,86,86,79,83,83,88,34,38,38,41,30,37,41,35,65,67,76,76,69,63,63,78]
# d_height = [25,28,19,30,21,22,17,35,56,57,50,53,60,53,49,61,22,25,19,30,21,24,28,18,50,50,50,48,50,50,45,51]
#
# d = zip(d_length, d_height) #시험문제
# l = list(d) #시험문제
# data = [list(x) for x in l] #시험문제
#
# #입력데이터셋
# X = np.array(data) #시험문제
# km = KMeans(n_clusters=4) #시험문제
# km.fit(X) #시험문제
# y_km = km.labels_ #시험문제
#
# plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='s', edgecolors='black', label='cluster 1')  #시험문제(x,y)자리
# plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='o', edgecolors='black', label='cluster 2')  #시험문제(x,y)자리
# plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', marker='v', edgecolors='black', label='cluster 3')   #시험문제(x,y)자리
# plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1], s=50, c='green', marker='*', edgecolors='black', label='cluster 4')   #시험문제(x,y)자리
#
# #plot the centroids
# print(print(km.cluster_centers_))
# plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
#             s=250,marker='*',edgecolors='black',label='centroids')
# plt.grid()
# plt.xlabel('강아지 길이')
# plt.ylabel('강아지 높이')
# plt.title('강아지 크기와 길이(클러스터링)')
# plt.legend(loc='upper left')
# plt.show()
#
# #새로운 데이터셋으로 그래프 그리기
# data=['길이 45, 높이 34','길이 70, 높이 59','길이 40 높이 30','길이 80, 높이 20','길이 80, 높이 50']
#
# #입력 데이터셋
# X1 = np.array([[45,34],[70,59],[40,30],[80,20],[80,50]])
# y_pred = km.predict(X1)
#
# plt.scatter(X[:,0],X[:,1],c=y_km, cmap='gist_rainbow', edgecolors='k', s=50)#시험문제
# plt.scatter(X1[:,0],X1[:,1],c=y_pred, cmap='gist_rainbow', edgecolors='k', s=50)#시험문제
#
# for i in range(5):
#     plt.text(X1[i,0], X1[i,1], data[i], c='green')
# plt.xlabel('Length', fontsize=15, c='orange')
# plt.ylabel('Height', fontsize=15, c='green')
# plt.title('테스트 데이터 클러스터링')
# plt.show()
#
# #문제 3. 고등학생 신장과 체중
# X = np.array([[160,55],[164,63],[165,50],[170,80],[175,73],[180,70],[155,43]])
# km = KMeans(n_clusters=2)
# km.fit(X)
# y_km = km.labels_
#
# X1 = np.array([[170,65]])
# y_pred = km.predict(X1)
#
# plt.scatter(X[y_km==0,0],X[y_km==0,1], s=50,c='lightgreen',marker='s',edgecolors='black',label='label 0')
# plt.scatter(X[y_km==1,0],X[y_km==1,1], s=50,c='orange',marker='o',edgecolors='black',label='label 1')
# plt.scatter(X1[0,0],X1[0,1],s=100,c='purple',marker='o',edgecolors='black',label='new data')
# plt.grid()
# plt.xlabel('신장')
# plt.ylabel('체중')
# plt.title('체중과 몸무게를 이용한 클러스터링')
# plt.legend(loc='upper left')
# plt.show()
#
# #문제4.
# iris = load_iris()
# X=iris.data
# y=iris.target
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
#
# X = X_test[:,2:]    #시험문제
# y_petal = y_test    #시험문제
#
# km = KMeans(n_clusters=3) #중심점 3개를 잡음.
# km.fit(X)
# labels = km.labels_ #fit의 결과값.
# centers = km.cluster_centers_
#
# X1 = np.array([[2.0,0.5],[4.7,1.4],[6.0,2.5],[4.0,1.7],[3.0,1.2]])
# y_pred = km.predict(X1)
#
# data = ['길이 2.0, 너비 0.5','길이 4.7, 너비 1.4','길이 6.0, 너비 2.5','길이 4.0, 너비 1.7','길이 3.0, 너비 1.2']
# #시각화
# plt.scatter(X[:,0],X[:,1],c=labels,cmap='gist_rainbow',edgecolors='k',s=50) #시험문제
# plt.scatter(X1[:,0],X1[:,1],c=y_pred,cmap='gist_rainbow',edgecolors='k',s=50) #시험문제
# for i in range(5):
#     plt.text(X1[i,0],X1[i,1],data[i],color='green')
# plt.xlabel('꽃잎 길이', fontsize=15, c='orange')
# plt.ylabel('꽃잎 너비', fontsize=15, c='green')
# plt.title('꽃잎의 길이와 너비의 클러스터링(KMeans)', fontsize=20, c='purple')
# plt.show()

#plt.scatter(c=labels, cmap='rainbow) -> labels에는 데이터 3개가 들어있음. rainbow 팔레트에서 lables 개수만큼 색을 쓴다는 의미.