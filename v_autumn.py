#step 0. 필요한 모듈과 라이브러리를 로딩하고 검색어를 입력 받습니다.
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd

# 검색할 팡ㄹ명 입력
query_txt = '가을여행'
f_name = 'Data/네이버_가을여행_블로그_뉴스.txt'
fc_name = 'Data/네이버_가을여행_블로그_뉴스.csv'

# step 1. 크롬 드라이버를 사용해서 웹 브라우저를 실행합니다.
path = 'C:/Temp/chromedriver.exe'
driver = webdriver.Chrome(path)
driver.get('http://www.naver.com')
driver.maximize_window()

# step 2. 네이버 검색창에 입력 받은 검색어를 넣고 검색
ele = driver.find_element_by_id('query')
ele.send_keys(query_txt)
ele.submit()

#블로그 탭으로 이동
driver.find_element_by_link_text('VIEW').click()
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
view_list = soup.find('ul','lst_total_list_base').find_all('li')

# step 2. 저장 목록을 만든 후, 목록에 있는 내용을 파일에 저장하기
no2 = []
title2=[]
contents2=[]
no=1

# 화면출력과 파일 저장을 동시에
for i in view_list:
    f = open(f_name, 'a',encoding='UTF-8')
    
    print(no)
    no2.append(no)
    f.write('1. 번호 : ' + str(no) + '\n')
    
    title = i.find('a','api_txt_lines total_tit _cross_trigger').get_text()
    title2.append(title)
    
    f.write('2. 제목' + str(title) + '\n')
    print(title)
    
    contents = i.find('div','api_txt_lines dsc_txt').get_text()
    contents2.append(contents)
    f.write('3. 내용'+str(contents)+'\n')
    print(contents)
    
    f.write('='*80+'\n')
    f.close()
    
    print('\n')
    no += 1

# 출력 결과를 txt 파일로 저장하기
print('txt 파일 저장 경로 : %s' %f_name)

# 출력 결과를 표(데이터 프레임)형태로 만들기
naver_result = pd.DataFrame()
naver_result['번호'] = no2
naver_result['제목'] = title2
naver_result['내용'] = contents2

# csv 형태로 저장하기
naver_result.to_csv(fc_name, encoding='utf-8-sig', index=False)
print('csv 파일 저장 경로 : %s'% fc_name)

# 확인차원
df = pd.read_csv('Data/네이버_가을여행_블로그_뉴스.csv', index_col=0)
print(df)
print(df['제목'])
print(df['내용'])