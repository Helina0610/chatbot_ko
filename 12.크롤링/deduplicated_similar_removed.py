#import json
#from sentence_transformers import SentenceTransformer, util

#input_path = "text_classification/datasets_drop_duplicates/deduplicated_dataset_주의.jsonl"
#output_path = "text_classification/deduplicated_similar_removed/deduplicated_dataset_주의.jsonl"

## ✅ 유사도 임계값
#SIMILARITY_THRESHOLD = 0.9

## ✅ 모델 로드 (한국어 지원 모델)
#model = SentenceTransformer("local_model/multilingual-e5-small-ko")

## ✅ 데이터 로딩
#data = []
#with open(input_path, "r", encoding="utf-8") as f:
#    for line in f:
#        data.append(json.loads(line))

## ✅ 중복 제거
#deduped = []
#embeddings = []

#for item in data:
#    text = item["text"].strip()
#    embedding = model.encode(text, convert_to_tensor=True)

#    is_duplicate = False
#    for existing_emb in embeddings:
#        sim = util.pytorch_cos_sim(embedding, existing_emb).item()
#        if sim >= SIMILARITY_THRESHOLD:
#            is_duplicate = True
#            break

#    if not is_duplicate:
#        deduped.append(item)
#        embeddings.append(embedding)

## ✅ 저장
#with open(output_path, "w", encoding="utf-8") as f:
#    for item in deduped:
#        f.write(json.dumps(item, ensure_ascii=False) + "\n")

#print(f"✅ 완료: {len(data)} → {len(deduped)}개로 중복 제거")

#import json
#import numpy as np
#import faiss
#from tqdm import tqdm
#from sentence_transformers import SentenceTransformer, util

#INPUT_PATH = "text_classification/datasets_drop_duplicates/deduplicated_dataset_시정(금액).jsonl"
#OUTPUT_PATH = "text_classification/deduplicated_similar_removed/deduplicated_dataset_시정(금액).jsonl"
#SIMILARITY_THRESHOLD = 0.9

## ✅ 1. 데이터 불러오기
#with open(INPUT_PATH, "r", encoding="utf-8") as f:
#    data = [json.loads(line.strip()) for line in f]

#texts = [item["text"].strip() for item in data]

## ✅ 2. SBERT 임베딩
#model = SentenceTransformer("local_model/multilingual-e5-small-ko")
#embeddings = model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True)

## ✅ 3. FAISS 인덱스 생성 (Cosine 유사도 → L2 거리 기반)
#dimension = embeddings.shape[1]
#index = faiss.IndexFlatIP(dimension)
#faiss.normalize_L2(embeddings)  # cosine similarity를 위해 L2 정규화
#index.add(x=embeddings)

## ✅ 4. 유사도 검색 및 중복 제거
#_, I = index.search(embeddings, 10)  # 각 문장마다 top-10 유사 문장 검색

#seen = set()
#deduplicated = []

#for i, neighbors in enumerate(I):
#    if i in seen:
#        continue

#    deduplicated.append(data[i])
#    for j in neighbors:
#        if i == j:
#            continue
#        score = np.dot(embeddings[i], embeddings[j])  # cosine similarity
#        if score >= SIMILARITY_THRESHOLD:
#            seen.add(j)

## ✅ 5. 저장
#with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#    for item in deduplicated:
#        f.write(json.dumps(item, ensure_ascii=False) + "\n")

#print(f"✅ 완료: {len(data)} → {len(deduplicated)}개 (중복 제거됨)")

import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_PATH = "text_classification/deduplicated_similar_removed/deduplicated_dataset_통보권고.csv"
OUTPUT_PATH = "text_classification/deduplicated_similar_removed/deduplicated_dataset_통보권고.csv"
SIMILARITY_THRESHOLD = 0.9

# ✅ 1. 데이터 불러오기 (CSV → DataFrame)
df = pd.read_csv(INPUT_PATH, encoding="utf-8")
assert "text" in df.columns, "CSV 파일에는 'text' 열이 있어야 합니다."

texts = df["text"].astype(str).str.strip().tolist()

# ✅ 2. SBERT 임베딩
model = SentenceTransformer("local_model/multilingual-e5-small-ko")
embeddings = model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True)

# ✅ 3. FAISS 인덱스 생성 (Cosine 유사도 기반 → L2 정규화)
dimension = embeddings.shape[1]
faiss.normalize_L2(embeddings)  # cosine similarity를 위해 L2 정규화
index = faiss.IndexFlatIP(dimension)
index.add(x=embeddings)

# ✅ 4. 유사도 검색 및 중복 제거
_, I = index.search(embeddings, 10)  # 각 문장마다 top-10 유사 문장 검색

seen = set()
deduplicated_rows = []

for i, neighbors in enumerate(I):
    if i in seen:
        continue
    deduplicated_rows.append(df.iloc[i].to_dict())
    for j in neighbors:
        if i == j:
            continue
        score = np.dot(embeddings[i], embeddings[j])  # cosine similarity
        if score >= SIMILARITY_THRESHOLD:
            seen.add(j)

# ✅ 5. 저장 (CSV)
dedup_df = pd.DataFrame(deduplicated_rows)
dedup_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"✅ 완료: {len(df)} → {len(dedup_df)}개 (중복 제거됨) → {OUTPUT_PATH}")
