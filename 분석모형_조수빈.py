import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
# font_location = 'C:/Windows/Fonts/HMFMPYUN.TTF'
# font_name = fm.FontProperties(fname=font_location).get_name()
# matplotlib.rc('font',family=font_name)
# font0 = {'family':'font_name',
#          'color':'red',
#          'weight':'normal',
#          'size':13}
# months = [str(x)+"월" for x in range(1,13)]
# months_int = [x for x in range(1,13)]
# w = pd.read_csv('Data/weather.csv', encoding='CP949')
# w['month'] = pd.DatetimeIndex(w['일시']).month
# means = w.groupby('month').mean()
# temp = means['평균기온(°C)']
# plt.plot(months, temp)
# plt.title('매월 평균기온의 평균')
# for i in range(0,12):
#     print(months_int[i])
#     plt.text(months_int[i], temp[i+1], round(temp[i+1],2))
#
# # print('결과창')
# # print(month_int)
# # print(temp)
# # print(temp[2])
# plt.show()

df = pd.DataFrame({
    'name':['Aa','Bb','Cc','Hh','Tt','Kk','Ww'],
    'hp':[230,250,190,300,210,220,270],
    'weight':[1.9,2.6,2.2,2.9,2.4,2.3,2.2],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2]
})
print(df)
df['performance'] = df['efficiency'] * df['hp']
print(df[df['performance'] == df['performance'].max()])