from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn import linear_model
import seaborn as sns

#머신러닝 선형회귀 연습!

regr = linear_model.LinearRegression()
# X = [[164],[179],[162],[170],[175]]
# y = [53,63,55,59,62]
# regr.fit(X,y)
# coef = regr.coef_
# intercept = regr.intercept_
# print('학습을 통해 얻어진 선형회귀직선의 방정식은')
# print('y= ', coef, '*X + ',intercept)
# print('---------------------------------\n')
#
# # 적합도 계산
# score = regr.score(X,y)
# print('The score is ', score)
# print()
#
# #test 데이터 삽입
# input_data = [[100],[185]] #무조건 2차원
# result = regr.predict(input_data) #predict : 문제집에 대한 답을 예측함
# print('result is ', result)
# print()
#
# #산점도 그리기
# print('학습데이터 x와 y를 이용한 산점도')
# plt.scatter(X, y, color='green', marker='*')
#
# y_pred = regr.predict(X)
# plt.scatter(X, y_pred, color='red')
#
# plt.plot(X, y_pred, color='blue', linewidth=3)
# plt.title('LinearRegression')
# plt.show()

# #테스트1
# X = [[164, 1], [167, 1], [165, 0], [170, 0], [163, 1], [159, 0], [166, 1]]
# y = [43, 48, 47, 67, 50, 52, 44]
# regr.fit(X,y)
# print('적합도 : ', regr.score(X,y))
#
# input_data = [[166, 1], [166, 0]]
# print('input_data의 추정 몸무게 : ', regr.predict(input_data))
# print('======================================\n')
#
# #데이터프레임에서 데이터 추출 후, 선형회귀
# df = pd.DataFrame({
#     'name' : ['A','B','C','D','E','F','G'],
#     'horse power' : [130,250,190,300,210,220,170],
#     'weight' : [1900, 2600, 2200, 2900, 2700, 2300, 2100],
#     'efficiency' : [16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2]
# })
# X = df[['horse power', 'weight']]
# print(X)
# y = df['efficiency']
# regr.fit(X,y)
# coef = regr.coef_
# intercept = regr.intercept_
# print('계수 : ', coef)
# print('절편 : ', intercept)
#
# result = regr.predict([[270,2500]])
# print('270 마력 2500kg 자동차의 예상 연비: {0:.2f}'.format(result[0]),'km/l')
# print()
#
# sns.pairplot(df[['horse power','weight','efficiency']])
# plt.show()
#
# print(df.corr())
# sns.heatmap(df.corr(), annot=True, cmap='BuGn_r', linewidths=2) #df.corr() 외우면 된다
# plt.show()
#
# #전체 데이터에서 훈련, 테스트용 데이터 구분
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)#외우자
# regr.fit(X_train, y_train)
# score = regr.score(X_train,y_train)
# print('test score is ', score)
# result = regr.predict([[270,2500]])
# print('270 마력 2500kg 자동차의 예상 연비: {0:.2f}'.format(result[0]), 'km/l')

from sklearn.datasets import load_iris
from sklearn import metrics
iris = load_iris()
from sklearn.neighbors import KNeighborsClassifier
X = iris.data
y = iris.target

# X_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
# knn = KNeighborsClassifier(n_neighbors=5) #기억!
# knn.fit(X,y)
#
# classes = {0:'setosa',1:'bersicolor',2:'virginica'}
# x_new = [[3,4,5,2],[5,4,2,2]]
# y_predict = knn.predict(x_new)
# print(classes[y_predict[0]])
# print(classes[y_predict[1]])
#
# y_predict_all = knn.predict(iris.data)
# scores = metrics.accuracy_score(iris.target, y_predict_all)

# #혼동행렬
# #머신의 정확성을 측정함.
# #진짜 참, 진짜 거짓, 가짜참(거짓인데 참으로 판단한거), 가짜거짓(참인데 거짓으로 판단.)
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# print('confusion matrix\n')
# cont_mat = confusion_matrix(iris.target, y_predict_all)#외울 부분
# print(cont_mat)
# plt.matshow(cont_mat)
# plt.show()

#강아지 문제집 만들기
print('dog\n')
d_length = [77,78,85,83,79,77,73,80]
d_height = [25,28,19,30,21,22,17,35]

samo_length = [75,77,86,86,79,83,83,88]
samo_height = [56,57,50,53,60,53,49,61]

malte_length = [34,38,38,41,30,37,41,35]
malte_height = [22,25,19,30,21,24,28,18]

dachhund = zip(d_length, d_height)
d_l = list(dachhund)
X1 = [list(i) for i in d_l]
y1 = [0] * len(X1)

s = zip(samo_length, samo_height)
print(s)
s_l = list(s)
X2 = [list(i) for i in s_l]
y2 = [1] * len(X2)

m = zip(malte_length, malte_height)
m_l = list(m)
X3 = [list(i) for i in m_l]
y3 = [2] * len(X3)

#전체 문제집
dogs = X1 + X2 + X3 # 수행평가때 비워질 부분
labels = y1 + y2 + y3   # 수행평가때 비워질 부분

print('neighbors의 개수 -> ', 5)
knn = KNeighborsClassifier(n_neighbors=5)# 수행평가때 비워질 부분
knn.fit(dogs, labels)# 수행평가때 비워질 부분

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

font_location = 'C:/Windows/Fonts/HMFMPYUN.TTF'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font',family=font_name)

x = [45,70,49,60]
y = [34,59,30,56]

