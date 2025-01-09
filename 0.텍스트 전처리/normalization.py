# 텍스트 정규화 및 기본 처리

import re

def normalize_text(text):
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    return text
