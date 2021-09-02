#step 1.
from konlpy.tag import *
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from wordcloud import WordCloud
from collections import Counter
import re
#
# # step 2.
# okt = Okt()
# data1 = open('Data/네이버_가을여행_블로그_뉴스.txt').read() # error 발생이면 txt파일 형식을 ANCI 로 변경
# print('\n')
#
# # step 3.
# data1 = re.sub("[0-9]+", '', data1)
# data2 = okt.nouns(data1)
# print('1.명사추출 키워드 : ', data2)
#
# # step 4. 불용어 1차 정리
# data3 = []
# for a in data2 :
#     if a=='제목':
#         data3.append(a.replace("제목", ' '))
#     elif a=='내용':
#         data3.append(a.replace('내용', ' '))
#     elif a=='번호':
#         data3.append(a.replace('번호', ' '))
#     else:
#         data3.append(a)
#
# # 글자수로 불용어 정리
# data4 = []
# for i in data3:
#     if len(i) >= 2 and len(i) <= 5:
#         data4.append(i)
#
# # step 5.
# print('단어의 빈도수 확인')
# data5 = Counter(data4)
# print('상위 100개 단어 추출')
# data6 = data5.most_common(100)
#
# print('2.단어별 빈도수:', data6)
# print()
#
# for i in enumerate(data6):
#     print(i)
#
# # step 6. 파일을 이용하여 불용어 제거하기
# sword = open('Data/가을여행블용어.txt').read()
# data7 = [each_word for each_word in data3
#          if each_word not in sword]
# data6_1=[]
# for a in data7:
#     if a=='가을':
#         data6_1.append(a.replace('가을','가을여행'))
#     elif a == '여행':
#         data6_1.append(a.replace('여행', ''))
#     elif a == '삼년산성':
#         data6_1.append(a.replace('삼년산성', '산성'))
#     elif a == '여행자':
#         data6_1.append(a.replace('여행자', ''))
#     else:
#         data6_1.append(a)
# # 글자수로 불용어 제거하기 -2차
# data8 = []
# for i in data6_1:
#     if len(i) >= 2 and len(i) <= 5:
#         data8.append(i)
#
# # step 7. 최종 단어별 빈도수 집계하기
# print()
# print('최종확인')
# data9 = Counter(data8)
#
# data10 = data9.most_common(45)
# print(data10)
#
# word = dict(data10)
#
# # step 8. 워드 클라우드 그리기
# wordcloud = WordCloud(font_path='C:/Windows/Fonts/HMFMPYUN.TTF', relative_scaling=0.4, background_color='white').generate_from_frequencies(word)
# plt.figure(figsize=(10,8))
# plt.imshow(wordcloud)
# plt.title('Autumn Tirp', fontsize=40)
# plt.axes('off')
# plt.show()

# s='010.1234.5678' #. -> -
# print('-'.join(s.split('.')))# 방법 1
# print(s.replace('.','-'))   # 방법 2
#
# #문제
# import re
# text='''101 COM Python
# 102 MAT Linear
# 103 ENG English'''
# s = re.findall('\d+',text)
# print(s) # 101 102 103  수숫자만 출력
# s = re.findall('[0-9]+', text)
# print(s)
#
# # 세계인권선언문
# f = open('Data/UNDHR.txt')
# for line in f:
#     if re.search('^\([0-9]+\)', line):
#         print(line)
#
# # 실전
# import re
#
# y='한글 텍스트 처리는 재밌다. 열심히 해야지ㅎㅎㅎㅎ, 맛있는 것도 먹고'
# y=re.sub('\.','',y)
# print(y)
# p = re.compile('[0-9]+')
# print(p.sub('', '10과 20의 합은 30입니다.'))
#
# s='정부가 발표하는 물가상승률과 소비자가 느끼는#### 물가상승률은 다르다####.'
# print('물가상승률:',str(s.count('물가상승률')))
# b = re.sub('\#', '',s)
# print(b)
#
# r = re.compile('a{2,3}b{2,3}') #a{num1,num2} : num1: 최소, num2 : 최대
# print(r.findall('aabb, aaabb, ab, aab, aaaaaaaaaaaabbbbbbbbbbbbbbbb'))
# print()
#
# p = re.compile('.+:')
# m = p.search('http://google.com')
# print(p.search('http://google.com'))
# print(m.group())
#
# print()
# p = re.compile('(내|나의|내꺼)')
# print(p.sub('그의','나의 물건에 손대지 마시오. 내 물건이요'))
# text1 = '나의 물건에 손대지 마시오.'
# text1 = re.sub('(내|나의|내꺼)','그의',text1)
# print(text1)

# 여러 문서(문장)의 유사률 구할 때 사용하는 것.
from konlpy.tag import Hannanum
from konlpy.tag import Kkma
from konlpy.tag import *
import re
text = '추석추석, 친척들이 모인 추석에서는 집값, 주식이 화제에 올랐다. 추석'
hannanum = Hannanum()
print(hannanum.morphs(text))
print(hannanum.nouns(text))
print()

kkma = Kkma()
print(kkma.morphs(text))
print(kkma.nouns(text))
print()

okt = Okt()
print(okt.morphs(text))
print(okt.nouns(text))
print()

original_data = '물가상승률에 대하여 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.'

# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.
token = re.sub("(\.)", '', original_data)
print(token)
token = okt.morphs(token)
print('불용어 제거')
data_set = []
for i in token:
    if len(i) > 1 :
        data_set.append(i)
print(data_set)
print()

print('불용어 제거 2')
sword=['하는','대하']
token = [each_word for each_word in data_set
         if each_word not in sword]
print(token)
print()

# 각 단어에 인덱스 부여
word2index = {}
bow=[]  # 단어 카운트

for voca in token:
    if voca not in word2index.keys():
        word2index[voca]=len(word2index)
        bow.insert(len(word2index)-1,1)
    else:
        index=word2index[voca]
        bow[index]=bow[index]+1

# 각 단어와 인덱스 번호
print()
print('각 단어와 각 단어에 해당되는 인덱스 번호')
print(word2index)
print()

for i in word2index:
    print(i, word2index[i])
print()
print('각 단어의 빈도수 \n')
print(bow)

print('단순히 빈도수만 구하는 것')
print('==============================')
print('token에는 불용어까지 제거된 결과가 들어 있다.')
print(token)
print()

print(Counter(token))
data = Counter(token)

for i in data:
    print(i, data[i])
print()
print('\n'*3)

print('first : bag of words end ==== korean')
print('\n'*3)

# 각 단어에 고유한 정수 인덱스를 부여한다.
original_data = '물가상승률에 대하여 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.'
word_dict={}
words_cnt_list=[]

for word in token:
    if word not in word_dict:
        word_dict[word] = len(word_dict)    # 각 단어에 고유한 정수 인덱스 부여
print()

print(word_dict)

print()
word_cnts = []
for word in word_dict.keys():
    word_cnts.append(original_data.count(word))
print(word_cnts)
print('두번째 : bag of words end ==== korean') #두번째 과정을 시험문제로 날 것이다