import re
from PyKomoran import Komoran
from konlpy.tag import Okt
from pykospacing import Spacing
from kiwipiepy import Kiwi

text = '안녕하세요 반갑습니다🐶'
text1 = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
text2 = "김형호영화시장분석가는'1987'의네이버영화정보네티즌10점평에서언급된단어들을지난해12월27일부터올해1월10일까지통계프로그램R과KoNLP패키지로텍스트마이닝하여분석했다."
text3 = "미리 예약을 할 수 있는 시스템으로 합리적인 가격에 여러 종류의 생선, 그리고 다양한 부위를 즐길 수 있기 때문이다 계절에 따라 모둠회의 종류는 조금씩 달라지지만 자주 올려주는 참돔 마스까와는 특히 맛이 매우 좋다 제철 생선 5~6가지 구성에 평소 접하지 못했던 부위까지 색다르게 즐길 수 있다"


# 텍스트 정규화 및 기본 처리
def normalize_text(text):
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 불용어 리스트 불러오기  
def read_stop_words_file():
    f= open("data/stop_words.txt",'r',encoding='utf-8')
    data = f.read()
    f.close()
    return data
  
# 형태소 분석 및 불용어 제거(Okt)
def morph_analysis_text(text):
    okt = Okt()

    stop_words = set(read_stop_words_file().split(' '))
    word_tokens = okt.morphs(text)

    result = [word for word in word_tokens if not word in stop_words]

    print('불용어 제거 전 :',word_tokens)
    print('불용어 제거 후 :',result)
  
 
# 형태소 분석 및 불용어 제거(Okt)
def morph_analysis_text2():
    okt = Okt()
    komoran = Komoran("STABLE")  # PyKomoran 초기화

    # 불용어 읽기 및 리스트화
    stop_words = set(read_stop_words_file().split(' '))

    # Okt를 사용한 형태소 분석 및 불용어 제거
    word_tokens_okt = okt.morphs(text1)
    result_okt = [word for word in word_tokens_okt if word not in stop_words]

    print("=== Okt 결과 ===")
    print('불용어 제거 전 (Okt):', word_tokens_okt)
    print('불용어 제거 후 (Okt):', result_okt)

    # PyKomoran을 사용한 형태소 분석 및 불용어 제거
    word_tokens_komoran = komoran.get_morphes_by_tags(text1)
    result_komoran = [word for word in word_tokens_komoran if word not in stop_words]

    print("=== PyKomoran 결과 ===")
    print('불용어 제거 전 (PyKomoran):', word_tokens_komoran)
    print('불용어 제거 후 (PyKomoran):', result_komoran)

    # PyKomoran 태그 기반 추가 필터링 (예: 명사만 추출)
    nouns_only = komoran.get_nouns(text1)
    
    print("=== PyKomoran 명사 추출 ===")
    print('명사만 추출 (PyKomoran):', nouns_only)
  
# 띄어쓰기 교정
def spacing_ko_text():
    spacing = Spacing()
    return spacing(text2)
    
## 한국어 문장 분리 도구 kss python3.12 지원 안됨
#def split_ko_text():
#    output = split_sentences(text3)
#    return output
    
# 한국어 문장 분리 도구 kiwi
def split_ko_text():
    kiwi = Kiwi()
    return  kiwi.split_into_sents(text3)
  
print(split_ko_text())