#from selenium import webdriver
#from selenium.webdriver.common.by import By
#from selenium.webdriver.chrome.service import Service
#from selenium.webdriver.chrome.options import Options
#import time
#import pandas as pd

## 크롬 옵션 설정
#chrome_options = Options()
## chrome_options.add_argument("--headless")
#chrome_options.add_argument("--disable-gpu")
#chrome_options.add_argument("--no-sandbox")
#chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

## 드라이버 실행
#service = Service()
#driver = webdriver.Chrome(service=service, options=chrome_options)

## 사이트 접속
#url = "https://www.bai.go.kr/bai/result/organ/list"
#driver.get(url)
#time.sleep(2)

## 결과 저장 리스트
#data = []

## 모든 tr 가져오기
#rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

#i = 0
#while i < len(rows) - 1:  # i+1 접근을 위해 -1
#    tr = rows[i]
#    try:
#        tds = tr.find_elements(By.TAG_NAME, "td")
#        if len(tds) < 4:
#            i += 1
#            continue

#        title_td = tds[0]
#        title = title_td.text.strip()
#        organization = tds[1].text.strip()
#        action_type = tds[2].text.strip()
#        document = tds[3].text.strip()

#        # 제목 클릭
#        driver.execute_script("arguments[0].scrollIntoView(true);", title_td)
#        title_td.click()
#        time.sleep(0.5)

#        # 다음 tr이 openCont 인지 확인
#        detail_text = ""
#        element = driver.find_element(By.CLASS_NAME, "openCont")
#        detail_text = element.text.split("\n")[2]
    

#        # 데이터 저장
#        data.append({
#            "제목": title,
#            "기관명": organization,
#            "조치구분": action_type,
#            "문서": document,
#            "세부내용": detail_text
#        })

#    except Exception as e:
#        print(f"[{i}] 오류 발생:", e)

#    i += 1

## 브라우저 종료
#driver.quit()

## 저장
#df = pd.DataFrame(data)
#df.to_csv("selenium_crawled_data.csv", index=False, encoding="utf-8-sig")
#print("크롤링 완료: selenium_crawled_data.csv")


#from selenium import webdriver
#from selenium.webdriver.common.by import By
#from selenium.webdriver.chrome.service import Service
#from selenium.webdriver.chrome.options import Options
#import time
#import pandas as pd

## 크롬 옵션 설정
#chrome_options = Options()
#chrome_options.add_argument("--disable-gpu")
#chrome_options.add_argument("--no-sandbox")
#chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

## 드라이버 실행
#service = Service()
#driver = webdriver.Chrome(service=service, options=chrome_options)

## 웹사이트 접속
#url = "https://www.bai.go.kr/bai/result/organ/list"
#driver.get(url)
#time.sleep(2)

## 결과 저장 리스트
#data = []
## 페이지 전체 반복
#page_count = 0
#max_pages = 10
#while True:
#    # 페이지 내 데이터 수집
#    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
#    i = 0
#    while i < len(rows) - 1:
#        tr = rows[i]
#        try:
#            tds = tr.find_elements(By.TAG_NAME, "td")
#            if len(tds) < 4:
#                i += 1
#                continue

#            title_td = tds[0]
#            title = title_td.text.strip()
#            organization = tds[1].text.strip()
#            action_type = tds[2].text.strip()
#            document = tds[3].text.strip()

#            # 제목 클릭
#            driver.execute_script("arguments[0].scrollIntoView(true);", title_td)
#            title_td.click()
#            time.sleep(0.5)

#            # openCont 클래스의 tr에서 세부내용 추출
#            detail_text = ""
#            try:
#                open_tr = driver.find_element(By.CLASS_NAME, "openCont")
#                detail_td = open_tr.find_element(By.CSS_SELECTOR, "td > div > table > tbody > tr > td")
#                detail_text = detail_td.text.strip()
#            except Exception as e:
#                print("세부내용 추출 실패:", e)

#            # 데이터 저장
#            data.append({
#                "제목": title,
#                "기관명": organization,
#                "조치구분": action_type,
#                "문서": document,
#                "세부내용": detail_text
#            })

#        except Exception as e:
#            print(f"[{i}] 오류 발생:", e)

#        i += 1

#    # 페이지 번호 클릭 시도
   
#    try:
#        page_links = driver.find_elements(By.CSS_SELECTOR, "ul.pages a")
#        current_page = None
#        for idx, a in enumerate(page_links):
#            if "on" in a.get_attribute("class"):
#                current_page = idx
#                break

#        # 다음 페이지 번호 클릭
#        if current_page is not None and current_page + 1 < len(page_links):
#            try:
#                next_page_link = page_links[current_page + 1]
#                driver.execute_script("arguments[0].click();", next_page_link)
#                time.sleep(1.5)
#                continue
#            except Exception as e:
#                print("다음 숫자 페이지 클릭 실패:", e)

#        # 숫자 페이지가 더 없으면 "다음 페이지" 버튼 클릭
#        next_btn = driver.find_element(By.CSS_SELECTOR, "ul.nextArea a.nextPage")
#        driver.execute_script("arguments[0].click();", next_btn)
#        time.sleep(2)

#    except Exception as e:
#        print("다음 페이지 없음 또는 에러:", e)
#        break

## 브라우저 종료
#driver.quit()

## 결과 저장
#df = pd.DataFrame(data)
#df.to_csv("selenium_crawled_data.csv", index=False, encoding="utf-8-sig")
#print("크롤링 완료: selenium_crawled_data.csv")

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 크롬 드라이버 설정
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")

service = Service()  # 또는 chromedriver 경로 지정
driver = webdriver.Chrome(service=service, options=chrome_options)

# 사이트 접속
url = "https://www.bai.go.kr/bai/result/organ/list"
driver.get(url)
time.sleep(2)

# select 박스 로딩까지 대기
select_element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "searchProc"))
)

# select에서 value='100' 신분적 사항, value='200' 금전적 사항, value='300' 주의통보등
select = Select(select_element)
select.select_by_value('100')

# 버튼이 클릭 가능해질 때까지 대기 후 클릭
search_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, ".searchForm button"))
)
search_button.click()
time.sleep(2)

# 결과 저장용 리스트
data = []

# 페이지 반복 제한
# ✅ 크롤링 시작 페이지 설정
start_page_num = 0  # ← 여기에 원하는 시작 페이지 번호를 설정하세요 (1부터 시작)

page_count = 0
max_pages = 160

while True:
    # 현재 페이지 번호 확인
    try:
        current_page_element = driver.find_element(By.CSS_SELECTOR, "ul.pages a.on")
        current_page_num = int(current_page_element.text.strip())
    except Exception as e:
        print("⚠️ 현재 페이지 번호 확인 실패:", e)
        break

    print(f"📄 현재 페이지: {current_page_num}")

    # 크롤링 시작 조건 검사
    if current_page_num < start_page_num:
        print(f"➡️ {start_page_num}페이지 전입니다. 다음 페이지로 이동합니다.")
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "ul.nextArea a.nextPage")
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(1.5)
            continue
        except Exception as e:
            print("⛔ 다음 페이지로 이동 실패:", e)
            break

    # ✅ 페이지 크롤링 시작
    print(f"✅ {current_page_num} 페이지 크롤링 시작")
    data = []

    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    i = 0
    while i < len(rows) - 1:
        tr = rows[i]
        try:
            tds = tr.find_elements(By.TAG_NAME, "td")
            if len(tds) < 4:
                i += 1
                continue

            title_td = tds[0]
            title = title_td.text.strip()
            organization = tds[1].text.strip()
            action_type = tds[2].text.strip()
            document = tds[3].text.strip()

            driver.execute_script("arguments[0].scrollIntoView(true);", title_td)
            title_td.click()
            time.sleep(0.5)

            detail_text = ""
            try:
                open_tr = driver.find_element(By.CLASS_NAME, "openCont")
                detail_td = open_tr.find_element(By.CSS_SELECTOR, "td > div > table > tbody > tr > td")
                detail_text = detail_td.text.strip()
            except Exception as e:
                print("   ⛔ 세부내용 추출 실패:", e)

            data.append({
                "label": action_type,
                "text": detail_text
            })

        except Exception as e:
            print(f"   ⛔ [{i}] 행 처리 오류:", e)
        i += 1

    # ✅ 페이지별 저장
    output_path = f"text_classification/output_1/신분적_page{current_page_num}.csv"
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"📁 저장 완료: {output_path}")

    page_count += 1
    if page_count >= max_pages:
        print("✅ 최대 페이지 도달, 종료합니다.")
        break

    # 다음 페이지로 이동
    try:
        next_btn = driver.find_element(By.CSS_SELECTOR, "ul.nextArea a.nextPage")
        driver.execute_script("arguments[0].click();", next_btn)
        time.sleep(2)
    except Exception as e:
        print("⛔ 다음 페이지 이동 실패:", e)
        break



# 브라우저 종료
driver.quit()

## 결과 저장
#df = pd.DataFrame(data)
#df.to_csv("text_classification/output/금전적사항.csv", index=False, encoding="utf-8-sig")
#print("📁 크롤링 완료: 금전적사항.csv")
