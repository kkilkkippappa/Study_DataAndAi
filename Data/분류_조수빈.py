from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
classes = {0:'sentosa',1:'versicolor',2:'virginica'}

X_new = [[3,4,5,2],[5,4,2,2]]
y_predict = knn.predict(X_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])

y_pred_all = knn.predict(iris.data)
scores = metrics.accuracy_score(iris.target, y_pred_all)
print('n_neighbors가 5일 때 정확도 : {0:.3f}'.format(scores))
print()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cont_mat = confusion_matrix(iris.target,y_pred_all)
print(cont_mat)
plt.matshow(cont_mat)
plt.show()

#문 2-2
d_length = [77,78,85,83,73,77,73,80,79]
d_height = [25,8,19,30,21,22,17,35,34]
s_length = [75,77,86,86,79,83,83,88,87]
s_height = [56,57,50,53,60,53,49,61,60]
m_length = [34,38,38,41,30,37,41,35,34]
m_height = [22,25,19,30,21,24,28,18,17]

d = zip(d_length,d_height)
l = list(d)
X1 = [list(i) for i in l]
y1 = [0]*len(l)

s = zip(s_length,s_height)
l = list(s)
X2 = [list(i) for i in l]
y2 = [1]*len(l)

m = zip(m_length,m_height)
l = list(m)
X3 = [list(i) for i in l]
y3 = [2]*len(l)

dogs = X1+X2+X3
labels=y1+y2+y3
# print(dogs,'\n',labels)
from sklearn.neighbors import KNeighborsClassifier
print('0:Dachshund, 1:Samoyed, 2:Maltese ')
dog_classes = {0:'Dachshund',1:'Samoyed',2:'Maltese'}
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(dogs,labels)

new_data = [[45,34],[70,59],[40,30],[80,20],[80,50]]
result = knn.predict(new_data)
print('길이 45, 높이 34: {}'.format(dog_classes[result[0]]))
print('길이 70, 높이 59: {}'.format(dog_classes[result[1]]))
print('길이 40, 높이 30: {}'.format(dog_classes[result[2]]))
print('길이 80, 높이 20: {}'.format(dog_classes[result[3]]))
print('길이 80, 높이 50: {}'.format(dog_classes[result[4]]))

import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()

font_location = 'C:/Windows/Fonts/HMFMPYUN.TTF'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font',family=font_name)

x = [45,70,40,80,80]
y=[34,59,30,20,50]
data=['길이 45, 높이 34','길이 70, 높이 59','길이 40, 높이 30','길이 80, 높이 20','길이 80, 높이 50']
plt.scatter(d_length,d_height,c='red',label='Dachshund')
plt.scatter(s_length,s_height,c='blue',marker='^',label='Samoyed')
plt.scatter(m_length,m_height,c='green',marker='s',label='Maltese')
plt.scatter(x,y,c='magenta',label='new Data')
for i in range(len(x)):
    plt.text(x[i],y[i],data[i],c='green')
plt.xlabel("Length")
plt.ylabel('Height')
plt.title('강아지의 길이와 높이에 따른 분류')
plt.legend(loc='upper left')
plt.show()