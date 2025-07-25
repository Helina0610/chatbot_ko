#import json

#input_path = "text_classification/datasets/preprocessed_고발.jsonl"   # 또는 preprocessed_all.jsonl 등
#output_path = "text_classification/datasets_drop_duplicates/deduplicated_dataset_고발.jsonl"

#seen_texts = set()
#deduplicated = []

#with open(input_path, "r", encoding="utf-8") as f:
#    for line in f:
#        data = json.loads(line)
#        text = data["text"].strip()

#        if text not in seen_texts:
#            seen_texts.add(text)
#            deduplicated.append(data)

## 저장
#with open(output_path, "w", encoding="utf-8") as f:
#    for item in deduplicated:
#        f.write(json.dumps(item, ensure_ascii=False) + "\n")

#print(f"✅ 중복 제거 완료: {len(deduplicated)}개 샘플")

import pandas as pd

input_path = "text_classification/split_by_label/징계문책.csv"
output_path = "text_classification/duplicates/징계문책_deduplicated.csv"

# CSV 로드
df = pd.read_csv(input_path, encoding="utf-8")

# 'text' 열 기준 중복 제거 (열 이름이 다르면 수정)
deduplicated_df = df.drop_duplicates(subset=["text"])  # 또는 subset=[df.columns[1]]로 첫 번째 라벨 제외 가능

# 저장
deduplicated_df.to_csv(output_path, index=False, encoding="utf-8-sig")  # Excel 호환 위해 utf-8-sig 권장

print(f"✅ 중복 제거 완료: {len(deduplicated_df)}개 샘플 저장됨 → {output_path}")
