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

#정답지랑 문제지 만들때 개수 조심하기

from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

regr = linear_model.LinearRegression()

# X = [[164], [179], [162], [170], [175]]  # 문제지와
# y = [53, 63, 55, 59, 62]  # 정답지 만들기
#
# regr.fit(X, y)  # 수행평가 할때 여기 비워짐 -> 학습시키는 함수 fit(문제지,정답지)
# coef = regr.coef_
# intercept = regr.intercept_
#
# print('학습을 통하여 구해진 선형 회귀 직선의 방정식은')
# print('y = ', coef, "* X + ", intercept)
# print('-----------------------------------------------------------------------------------------------')
#
# score = regr.score(X, y)  # 수행평가 할때 여기 비워짐 -> 적합도 계산 score(문제지, 정답지)
# print("The score of this line for the data: ", score)
#
# input_data = [[180], [185]]  # 문제지는 무조건 2차원으로 만들어야함
# result = regr.predict(input_data)  # 예측하는 함수 predict(문제지)
# print(result)
# print('-----------------------------------------------------------------------------------------------')
#
# print("학습 데이터 X와 y 데이터를 이용한 산점도")
# plt.scatter(X, y, color='green', marker='*')
#
# y_pred = regr.predict(X)  # 수행평가 할때 여기 비워짐
# plt.scatter(X, y_pred, color='red')
#
# plt.plot(X, y_pred, color='yellowgreen', linewidth=3)  # 수행평가 할때 여기 비워짐
#
# plt.title('Linearregression')
# plt.show()
# print('-----------------------------------------------------------------------------------------------')
#
# X = [[164, 1], [167, 1], [165, 0], [170, 0], [163, 1], [159, 0], [166, 1]]  # 수행평가 할때 여기 비워짐 -> 문제지
# y = [43, 48, 47, 67, 50, 52, 44]  # 수행평가 할때 여기 비워짐 -> 정답지
#
# regr.fit(X, y)  # 수행평가 할때 여기 비워짐
# print('적합도: ', regr.score(X, y))  # 수행평가 할때 여기 비워짐
#
# input_data = [[166, 1], [166, 0]]
# print('추정몸무게: ', regr.predict(input_data))
# print('-----------------------------------------------------------------------------------------------')
#
# df = pd.DataFrame({
#     'name' : ['A','B','C','D','E','F','G'],
#     'horse power' : [130,250,190,300,210,220,170],
#     'weight' : [1900, 2600, 2200, 2900, 2700, 2300, 2100],
#     'efficiency' : [16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2]
# })
#
# print(df)
# print(df.keys())
#
# X = df[['horse power', 'weight']] # 수행평가때 비워질 부분
# print(X)
#
# y = df['efficiency'] # 수행평가때 비워질 부분
# print(y)
#
# regr.fit(X, y) # 회귀모형 만들기 # 수행평가때 비워질 부분
# coef = regr.coef_
# intercept = regr.intercept_
# print('계수: ',coef)
# print('절편: ',intercept)
# print()
# score = regr.score(X,y) # 수행평가때 비워질 부분
# print('예측모델의 적합도 점수: ',score)
# print("예측하기")
#
# result = regr.predict([[270,2500]]) # 수행평가때 비워질 부분
# print('270 마력 2500kg 자동차의 예상 연비: {0:.2f}'.format(result[0]),'km/l')
# print()
#
# sns.pairplot(df[['horse power','weight','efficiency']])
# plt.show()
#
# print(df.corr())
# sns.heatmap(df.corr(), annot=True,cmap='YlGnBu',linewidths=2) # 수행평가때 비워질 부분 df.corr() # 색깔 9개짜리
# plt.show()
# print('-----------------------------------------------------------------------------------------------')
#
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) # test_size = 테스트 데이터 # 수행평가때 비워질 부분
# # X : 문제집
# # y : 답안지
# # train : 훈련 데이터, test : 테스트 데이터
#
# regr.fit(X_train,y_train) # 수행평가때 비워질 부분
# score = regr.score(X_train,y_train) # 수행평가때 비워질 부분
# print("예측모델의 적합도: ", score)
# result = regr.predict([[270, 2500]]) # 수행평가때 비워질 부분 predict([[새로운 문제, 새로운 문제]])
# print('270 마력 2500kg 자동차의 예상 연비: {0:.2f}'.format(result[0]), 'km/l')
# print('-----------------------------------------------------------------------------------------------')
################KNN
from sklearn.datasets import load_iris
from sklearn import metrics
iris = load_iris()
from sklearn.neighbors import KNeighborsClassifier
print('\n\n\n\n')
X = iris.data # 수행평가때 비워질 부분
y = iris.target # 수행평가때 비워질 부분

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)  # 수행평가때 비워질 부분(test_size)

knn = KNeighborsClassifier(n_neighbors=5)  # 수행평가때 비워질 부분 n_neightbors = 이웃 data 몇 개 이용
knn.fit(X,y) # 수행평가때 비워질 부분

classes = {0:'setosa', 1:'bersicolor', 2:'virginica'}
x_new = [[3, 4, 5, 2],   # 수행평가때 비워질 부분
         [5, 4, 2, 2]]

y_predict = knn.predict(x_new) # 수행평가때 비워질 부분
print(classes[y_predict[0]])
print(classes[y_predict[1]])

y_pred_all = knn.predict(iris.data)
scores = metrics.accuracy_score(iris.target, y_pred_all) # 분류에서의 score

#혼동행렬
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
print('\n\n\n\n')
#confusion_matrix : 틀린 부분을 지적.
conf_mat = confusion_matrix(iris.target, y_pred_all) # 수행평가때 비워질 부분
print(conf_mat)
plt.matshow(conf_mat)
plt.show()
print('-----------------------------------------------------------------------------------------------')

d_length = [77,78,85,83,79,77,73,80]
d_height = [25,28,19,30,21,22,17,35]

samo_length = [75,77,86,86,79,83,83,88]
samo_height = [56,57,50,53,60,53,49,61]

malte_length = [34,38,38,41,30,37,41,35]
malte_height = [22,25,19,30,21,24,28,18]

#문제지 만들기
print('dachhund---------')
# zip : 지퍼처럼 두 인자를 묶기.
dachhund = zip(d_length, d_height) #길이랑 키 엮어버리기 # 수행평가때 비워질 부분
d_l = list(dachhund)
X1 = [list(x) for x in d_l] # 수행평가때 비워질 부분
print(X1)

y1 = [0] * len(X1) # 0으로 레이블링 하기  # 수행평가때 비워질 부분
print(y1)
print()

print('samoyed---------')
samoyed = zip(samo_length,samo_height)
s_l = list(samoyed)
X2 = [list(x) for x in s_l]
print(X2)

y2 = [1] * len(X2)
print(y2)
print()

print('maltese---------')
maltese = zip(malte_length, malte_height)
m_l = list(maltese)
X3 = [list(x) for x in s_l]
print(X3)

y3 = [2] * len(X3)
print('y3 = ',y3)
print()

#전체 문제지 만들기
dogs = X1 + X2 + X3 # 수행평가때 비워질 부분
labels = y1 + y2 + y3 # 수행평가때 비워질 부분

print('neighbor의 갯수 => ', 5)
knn = KNeighborsClassifier(n_neighbors=5) # 수행평가때 비워질 부분
knn.fit(dogs, labels) # 수행평가때 비워질 부분 _학습시키기

new_data = [[45, 34], [70, 59], [49, 30], [80, 27]] # 수행평가때 비워질 부분
dog_classes = {0:'Dachshund', 1:'Samoyed', 2:'maltese'}

result = knn.predict(new_data) # 수행평가때 비워질 부분
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

x = [45, 70, 49, 60] # 수행평가때 비워질 부분
y = [34, 59, 30, 56] # 수행평가때 비워질 부분

data = ['길이 45, 높이 34', '길이 78, 높이 59', '길이 49, 높이 39', '길이 60, 높이 56']

plt.scatter(d_length,d_height, c='red', label='Dachshund')
plt.scatter(samo_length,samo_height, c='blue',marker='^', label='Samoyed')
plt.scatter(malte_length,malte_height, c='yellowgreen', marker='s', label='Maltese')
plt.scatter(x, y, c='orange', label='new Data') # 수행평가때 비워질 부분

for i in range(4):
    plt.text(x[i], y[i], data[i], color='green') # 수행평가때 비워질 부분

plt.xlabel('Length')
plt.ylabel('Height')
plt.title('Dog size')
plt.legend(loc='upper left')

plt.show()