from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
import numpy as np

font_location = 'C:/Windows/Fonts/HMFMPYUN.TTF'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font',family=font_name)

# X = np.array([
#     [6,3],[11,15],[17,12],[24,10],[20,25],[22,30],[85,70],[71,81],[60,79],[56,52],[81,91],[80,81]
# ])
# kmeans = KMeans(n_clusters=2).fit(X)
# y = kmeans.labels_
# centers = kmeans.cluster_centers_
#
# new = np.array([[10,10],[60,50]])
# y_pred = kmeans.predict(new)
# new_data = ['(10,10)','(60,50)']
# plt.scatter(X[:,0], X[:,1], c=y, edgecolors='orange', cmap='rainbow', s=50)
# plt.scatter(centers[:,0], centers[:,1], c='green', s=200, alpha=0.5)
# data='중심점'
# for i in range(2):
#     plt.text(centers[i,0], centers[i,1], new_data[i], color='green')
# plt.title('클러스터링 k=2', fontsize=20)
# plt.show()

# #문2. 아이리스 데이터
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=4)
# X = X_train[:,:2]
# y_sepal=y_train
# print('X_train 의 크기 : {}'.format(X_train.shape))
# print('y_train 의 크기 : {}'.format(y_train.shape))
# print('X 의 크기 : {}'.format(X.shape))
# print('y 의 크기 : {}'.format(y_sepal.shape))
#
# km = KMeans(n_clusters=3)
# km.fit(X)
# labels = km.labels_
# centers = km.cluster_centers_
#
# fig,axes = plt.subplots(1,2,figsize=(10,5))
# axes[0].scatter(X[:,0], X[:,1], c=y_sepal, cmap='gist_rainbow',edgecolor='k',s=50)
# axes[1].scatter(X[:,0], X[:,1], c=labels, cmap='rainbow',edgecolor='k',s=50)
# axes[0].set_xlabel('꽃받침 길이', fontsize=10, c='orange')
# axes[0].set_ylabel('꽃받침 너비', fontsize=10)
# axes[1].set_xlabel('꽃받침 길이', fontsize=10, c='orange')
# axes[1].set_ylabel('꽃받침 너비', fontsize=10)
# axes[0].set_title('붓꽆의 기본 클러스터링', fontsize=15, c='purple')
# axes[1].set_title('꽃받침의 길이와 너비의 클러스터링(KMeans)', fontsize=15, c='purple')
# axes[1].scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
#                 s=250, marker='*', c='yellow', edgecolor='black', label = 'centroids')
# plt.show()
#
# X1 = np.array([[5.4,5.2],[5.7,3.4],[6.0,2.5],[6.3,3.7],[5.7,4.2]])
# y_pred = km.predict(X1)
# plt.scatter(X[:,0],X[:,1],c=labels,cmap='gist_rainbow',edgecolors='k',s=50)
# plt.scatter(X1[:,0],X1[:,1],c=y_pred,cmap='rainbow',edgecolors='k',marker='*',s=300)
# plt.xlabel('Length',fontsize=10,c='orange')
# plt.ylabel('Height',fontsize=10,c='green')
# plt.title('테스트 데이터', fontsize=15, c='purple')
# plt.show()
## 몸무게 체중 문제
# X1 = np.array([[5.4,5.2],[5.7,3.1],[6.0,2.4],[6.3,8.4],[5.1,3.0]])
# y_pred = km.predict(X1)
#
# plt.scatter(X[:,0], X[:,1], c=labels, cmap='gist_rainbow')
# plt.scatter(X1[:,0],X1[:,1],c=y_pred, cmap='rainbow',edgecolors='k',marker='*')

# #다른 문제
# X = np.array([
#     [160,55],[164,63],[165,50],[170,80],[175,73],[180,70],[155,43]
# ])
# km = KMeans(n_clusters=2)
# km.fit(X)
# y_km = km.labels_
# X1 = np.array([[170,65]])
# y_pred = km.predict(X1)
# print(y_km)
# print(X[y_km==0,0])
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

#문4. 강아지
d_length = [77,78,85,83,73,77,73,80,75,77,86,86,79,83,83,88,34,38,38,41,30,37,41,35,65,67,76,76,69,63,63,78]
d_height = [25,28,19,30,21,22,17,35,56,57,50,53,60,53,49,61,22,25,19,30,21,24,28,18,50,50,50,48,50,50,45,51]

d = zip(d_length, d_height) #시험문제
l = list(d) #시험문제
data = [list(x) for x in l] #시험문제

X = np.array(data)
km = KMeans(n_clusters=4)
km.fit(X)
y_km = km.labels_