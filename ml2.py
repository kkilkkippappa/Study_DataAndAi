
#정답지랑 문제지 만들때 개수 조심하기

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris= load_iris()

print('iris의 keys \n{}'.format(iris.keys()))
print()

print(type(iris.data))
print('iris의 data 크기 \n{}'.format(iris['data'].shape))
print(iris.data[:3,:])
print()

print('feature names')
print(iris.feature_names)
print()


print('iris의 target names')
print('iris.target_names \n{}'.format(iris.target_names))
print('0=setosa, 1=versicolor, 2=virginica')
print()

print('iris의 target 크기 \n{}'.format(iris.target.shape))
print(type(iris.target))
print(iris.target[:6])
print()

print('iris의 DESCR \n{}'.format(iris['DESCR']))
print('-----------------------------------------------------------------------------------------------')

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df.head())

iris_df2 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names']+['target'])
print(iris_df2.head())
print('-----------------------------------------------------------------------------------------------')

X_train,X_test,y_train,y_test = train_test_split(iris['data'],iris['target']) #testsize가 없으면 훈련: 75%, 테스트: 25%로 분류

print('x_train의 크기: {}'.format(X_train.shape))
print('x_test의 크기: {}'.format(X_test.shape))
print('y_train의 크기: {}'.format(y_train.shape))
print('y_test의 크기: {}'.format(y_test.shape))

import seaborn as sns
print(['picture 2'])
sns.pairplot(iris_df2,
             diag_kind='kde',
             hue='target',
             palette='colorblind')
plt.show()
print('-----------------------------------------------------------------------------------------------')


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

num_neigh = 1
knn = KNeighborsClassifier(n_neighbors= num_neigh)
knn.fit(X_train, y_train) #학습시킬때 필요한 데이터를 대입

print('테스트 데이터를 이용하여 예측')
y_pred = knn.predict(X_test)

scores = metrics.accuracy_score(y_test, y_pred) #accuracy_score == 정확도 확인 (정답지, 예측모델이 만든 정답)

print('n_neighbors가 {0:d}일때 정확도: {1:.3f}'.format(num_neigh, scores))
print('-----------------------------------------------------------------------------------------------')

for i in range(1, 11):
    num_neigh = i
    knn = KNeighborsClassifier(n_neighbors=num_neigh)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    scores = metrics.accuracy_score(y_test, y_pred)
    print('n_neighbors가 {0:d}일때 정확도 : {1:.3f}'.format(num_neigh, scores))
print('-----------------------------------------------------------------------------------------------')

print('iris의 target names \n{}'.format(iris.target_names))
print('0=setosa, 1=versicolor, 2=virginica')

print('새로운 데이터 1')
X_new = [[5,2.9,1,0.2]]

prediction = knn.predict(X_new)
print('예측한 타깃의 이름: {}'.format(iris['target_names'][prediction]))
print('-----------------------------------------------------------------------------------------------\n')

num_neigh = 5
nkk = KNeighborsClassifier(n_neighbors=num_neigh)
knn.fit(iris.data, iris.target)
y_pred_all = knn.predict(iris.data)
scores = metrics.accuracy_score(iris.target, y_pred_all)
print('n_neighbors가 {0:d}일때 정확도 : {1:.3f}'.format(num_neigh, scores))

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(iris.target, y_pred_all)
print(conf_mat)
plt.matshow(conf_mat)
plt.show()

print('두 그룹의 데이터를 서로 엮어주는 파이썬의 내장 함수 zip')
numbers = [1,2,3]
letters= ['A','B','C']
print('zip()')
for pair in zip (numbers, letters):
    print(pair)

print('zip 후 list로 변환')
pairs = list(zip(numbers, letters))
print(pairs)

pairs_2 = [list(x) for x in pairs] #문제지 만들기
print(pairs_2)

#갱얼쥐
print('-----------------------------------------------------------------------------------------------\n')
d_length = [77,78,85,83,79,77,73,80]
d_height = [25,28,19,30,21,22,17,35]

samo_length = [75,77,86,86,79,83,83,88]
samo_height = [56,57,50,53,60,53,49,61]

malte_length = [34,38,38,41,30,37,41,35]
malte_height = [22,25,19,30,21,24,28,18]

print('\n닥스훈트-----------------------------------------')
#---- 문제지-----
dachshund = zip(d_length, d_height) #길이랑 키랑 엮기
l = list(dachshund)
X1 = [list(x) for x in l]
print(X1)

#---- 정답지-----
y1= [0] * len(X1)
print(y1)
print()

print('\n사모예드-----------------------------------------')
#---- 문제지-----
samoyed = zip(samo_length, samo_height)
l = list(samoyed)
X2 = [list(x) for x in l]
print(X2)

#---- 정답지-----
y2= [1] * len(X2)
print(y2)
print()

print('\n말티즈-------------------------------------------')
#---- 문제지-----
maltese = zip(malte_length, malte_height)
l = list(maltese)
X3 = [list(x) for x in l]
print(X3)

#---- 정답지-----
y3= [2] * len(X3)
print(y3)
print()

dogs = X1 + X2 + X3
labels = y1+y2+y3

print(dogs)
print(labels)

print('neighbor의 갯수 => ', 5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(dogs,labels)

new_data = [[45,34],[70,59],[49,30],[80,27]]
dog_classes = {0:'Dachshund', 1:'Samoyed', 2:'maltese'}

result = knn.predict(new_data)
print(' 길이 45, 높이 34: {}'.format(dog_classes[result[0]]))
print(' 길이 70, 높이 59: {}'.format(dog_classes[result[1]]))
print(' 길이 49, 높이 30: {}'.format(dog_classes[result[2]]))
print(' 길이 80, 높이 27: {}'.format(dog_classes[result[3]]))

#시각화
import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()

font_location = 'C:\Windows\Fonts\HanSantteutDotumRegular.ttf'

font_name = fm.FontProperties(fname = font_location).get_name()
matplotlib.rc('font', family = font_name)

x= [45,78,49,60]
y= [34,59,30,56]

data = ['길이 45, 높이 34', '길이 78, 높이 59', '길이 49, 높이 39', '길이 60, 높이 56']

plt.scatter(d_length,d_height, c='red', label='Dachshund')
plt.scatter(samo_length,samo_height, c='blue',marker='^', label='Samoyed')
plt.scatter(malte_length,malte_height, c='yellowgreen',marker='s' ,label='Maltese')
plt.scatter(x, y, c='magenta', label='new Data')

for i in range(4):
    plt.text(x[i], y[i], data[i], color='green')

plt.xlabel('Length')
plt.ylabel('Height')
plt.title('Dog size')
plt.legend(loc='upper left')

plt.show()