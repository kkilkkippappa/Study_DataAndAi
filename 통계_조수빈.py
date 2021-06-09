import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
font_location = 'C:/Windows/Fonts/HMFMPYUN.TTF'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font',family=font_name)

players = [[190,76.4],
           [185,86.2],
           [181,78.5],
           [190,88.5],
           [186,70.1]]
players_np = np.array(players)

list = list()
for i in players:
    if i[1] >= 80:
        list.append(i)
print('몸무게 80 이상', list)
for i in players:
    if i[0] >= 185:
        list.append(i)
print('키 185 이상', list)

num_array_1 = np.arange(1,51)
print(num_array_1)
num_array_2 = num_array_1 + 3
print(num_array_2.mean())
print(np.median(num_array_2))
print(np.corrcoef([num_array_1,num_array_2]))
num_array_2 = num_array_2.reshape(10,5)
print(num_array_2)

weather = pd.read_csv('Data/weather.csv', encoding='CP949')
print('평균 분석--------')
print(weather.mean())
print('최대값 분석--------')
print(weather.max())