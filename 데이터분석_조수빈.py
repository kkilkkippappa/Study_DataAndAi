import numpy as np
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
font_location = 'C:/Windows/Fonts/HMFMPYUN.TTF'
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font',family=font_name)
font0 = {'family':'font_name',
         'color':'purple',
         'weight':'bold',
         'size':16}
f = plt.figure(figsize=(8,8))
ax = f.add_subplot()
x = [str(x+1)+"월" for x in range(7,13)]
y = [456,492,578,599,670,854]
import seaborn as sns
colors = sns.color_palette('Set3', 6)
bars = plt.bar(range(len(x)), y, color = colors)
for i,b in enumerate(bars):
    ax.text(b.get_x() + b.get_width() * (1/2), b.get_height(), y[i], ha='center', fontdict=font0)
plt.title('신규가입자', fontsize=25)
plt.ylabel('가입자수', fontsize=15)
plt.xlabel('월', fontsize=15)
plt.show()

c = pd.read_csv('Data/countries.csv', index_col=0)
print(c)
c['area'].plot(kind='bar', color='orange')
plt.legend()
plt.show()
