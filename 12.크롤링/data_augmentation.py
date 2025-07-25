#import glob
#import pandas as pd

#file_names = glob.glob("text_classification/output_1/*.csv")
#total = pd.DataFrame()

#for file_name in file_names:
#    try:
#        tmp = pd.read_csv(file_name, encoding="utf-8")
#        total = pd.concat([total, tmp], ignore_index=True)
#    except pd.errors.EmptyDataError:
#        print(f"⚠️ 빈 파일 무시됨: {file_name}")

#total.to_csv("text_classification/result_datasets/신분적사항_total.csv", index=False, encoding="utf-8")
#print("📁 크롤링 완료: 신분적사항_total.csv")

import pandas as pd
import os

# 🔹 전체 데이터 로드
df = pd.read_csv("text_classification/result_datasets/신분적사항_total.csv", encoding="utf-8")

# 🔹 저장 경로 설정
output_dir = "text_classification/split_by_label"
os.makedirs(output_dir, exist_ok=True)

# 🔹 첫 번째 열 이름 가져오기
label_col = df.columns[0]  # 첫 번째 열이 라벨이라고 가정

# 🔹 라벨별로 분리하여 저장
for label, group in df.groupby(label_col):
    # 파일명에 특수문자 등이 있으면 안전하게 처리
    safe_label = str(label).strip().replace("/", "_").replace(" ", "_")
    save_path = os.path.join(output_dir, f"{safe_label}.csv")
    
    group.to_csv(save_path, index=False, encoding="utf-8")  # Excel 호환 위해 utf-8-sig 사용
    print(f"✅ 저장됨: {save_path}")
