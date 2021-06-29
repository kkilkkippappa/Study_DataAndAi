import numpy as np
from sklearn import linear_model
regr = linear_model.LinearRegression()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

X = [[164],[167],[165],[170],[179],[163],[159],[166]]
y = [43,48,47,66,67,50,52,44]
regr.fit(X,y)
coef = regr.coef_
intercept = regr.intercept_
print('몸무게 = ', coef,'* 키 + ', intercept)
score = regr.score(X,y)
print('선형회귀방정식의 적합도 : ', score)
print()

#키 : 185, 몸무게?
new_data = [[185]]
weight_pred = regr.predict(new_data)
print('키 185에 해당되는 예측 몸무게 \n{}', format(weight_pred))

#그래프
plt.scatter(X,y,c='green',marker='*')
y_pred = regr.predict(X)
plt.plot(X,y_pred,c='yellow',linewidth=3)
plt.show()

#문2
df = pd.DataFrame({
    'name':['A','B','C','D','E','F','G','H','I','K'],
    'horse power':[130,250,190,300,210,220,170,200,300,290],
    'weight':[1900,2600,2200,2900,2400,2300,2100,2300,2800,2700],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2,15.2,8.1,9.0]
})
X=df[['horse power','weight']]
y = df['efficiency']

regr.fit(X,y)
score = regr.score(X,y)
print('예측모델의 적합도 : ', score)
result = regr.predict([[270,2500],[300,3100]])
print('270 마력 2500kg 자동차의 예상 연비 : {0:.2f}'.format(result[0], 'km/l'))
print('300 마력 3100kg 자동차의 예상 연비 : {0:.2f}'.format(result[1], 'km/l'))

sns.heatmap(df.corr(), annot=True, cmap='YlGnBu',linewidths=2)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
regr.fit(X_train,y_train)
result = regr.predict([[300,3200]])
print('300 마력 3200kg 자동차의 예상 연비 : {0:.2f}'.format(result[0], 'km/l'))