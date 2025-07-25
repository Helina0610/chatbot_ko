import os
import pandas as pd
from tqdm import tqdm
import asyncio
from googletrans import Translator  # 비동기 버전 사용 가정

# 🔧 경로 설정
INPUT_PATH = "text_classification/duplicates/고발_deduplicated.csv"
OUTPUT_PATH = "text_classification/back_Translation/고발_augmented.csv"

# ✅ 증강 횟수 설정
AUGMENT_TIMES = 1  # 문장당 몇 번 증강할지

# ✅ 데이터 로드 (CSV)
df = pd.read_csv(INPUT_PATH, encoding="utf-8")
assert "label" in df.columns and "text" in df.columns, "CSV에는 'label'과 'text' 열이 있어야 합니다."

# ✅ ko2en / en2ko 함수 정의
async def ko2en(translator, text):
    outStr = await translator.translate(text=text, src='ko', dest='en')
    return outStr.text

async def en2ko(translator, text):
    outStr = await translator.translate(text=text, src='en', dest='ko')
    return outStr.text

# ✅ 메인 비동기 루프
async def main():
    translator = Translator()  # 단일 Translator 인스턴스 생성
    augmented_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = row["label"]
        text = str(row["text"]).strip()
        print(f"🔍 원본 문장: {text}")

        try:
            for _ in range(AUGMENT_TIMES):
                en = await ko2en(translator, text)
                ko = await en2ko(translator, en)
                augmented_rows.append({
                    "label": label,
                    "text": ko
                })
        except Exception as e:
            print(f"⚠️ 증강 실패: {text} - {e}")

    # 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    aug_df = pd.DataFrame(augmented_rows)
    aug_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"✅ 증강 완료: {len(augmented_rows)}개 문장 저장됨 → {OUTPUT_PATH}")

# 🔁 실행
asyncio.run(main())

