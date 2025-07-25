#import matplotlib.pyplot as plt
#from matplotlib import font_manager, rc

## 한글 폰트 설정 (나눔고딕 예시)
#font_path = "fonts/NanumGothic.ttf"  # 리눅스 경로 예시
#font = font_manager.FontProperties(fname=font_path).get_name()
#rc('font', family=font)

## 데이터
#categories = ['고발', '모범', '변상', '시정(금액)', '시정(기타)', '인사자료통보', '주의', '징계문책', '통보권고']
#values = [25, 114, 67, 521, 141, 229, 5231, 1139, 6233]

#plt.figure(figsize=(10,6))
#bars = plt.bar(categories, values, color='skyblue')
#plt.title('사건 유형별 건수')
#plt.ylabel('건수')
#plt.xticks(rotation=45)

## 값 표시
#for bar in bars:
#    yval = bar.get_height()
#    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, yval, ha='center', va='bottom')

#plt.tight_layout()
#plt.show()

from itertools import cycle, islice
from datasets import Dataset,load_dataset
import pandas as pd

labels = ["고발", "모범", "변상","시정(금액)","시정(기타)","인사자료통보","주의","징계문책","통보권고"]  # 원하는 라벨 리스트
file_paths = [f"text_classification/results/{label}.csv" for label in labels]

# 각 CSV를 개별 Dataset으로 불러옴
datasets = [load_dataset("csv", data_files=path, split="train") for path in file_paths]

# 각 데이터셋에 라벨 추가
for ds, label in zip(datasets, labels):
    ds = ds.map(lambda x: {**x, "label": label})


# zip_longest로 한 줄씩 라운드로빈
def interleave_datasets_round_robin(datasets, limit=None):
    iterators = [iter(ds) for ds in datasets]
    while True:
        for it in iterators:
            try:
                yield next(it)
            except StopIteration:
                iterators.remove(it)
                if not iterators:
                    return

# 예: 최대 1000개만 미리 만들어보기
interleaved = list(islice(interleave_datasets_round_robin(datasets), 10000))

output_path = f"text_classification/output/round_robin_interleaved_1000.csv"
df = pd.DataFrame(interleaved)
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"📁 저장 완료: {output_path}")

#📦output
# ┣ 📜round_robin_interleaved.csv
# ┣ 📜round_robin_interleaved_v1.csv 고발 25개에 맞춰서
# ┣ 📜round_robin_interleaved_v2.csv 변상 67개에 맞춰서
# ┣ 📜round_robin_interleaved_v3.csv 모범 114개에 맞춰서
# ┣ 📜round_robin_interleaved_v4.csv 시정(기타) 141개에 맞춰서
# ┣ 📜round_robin_interleaved_v5.csv 인사자료통보 229개에 맞춰서
# ┣ 📜round_robin_interleaved_v6.csv 시정(금액) 521개 맞춰서
# ┗ 📜round_robin_interleaved_v7.csv 징계문책 1139개 맞춰서


#final_dataset = Dataset.from_list(interleaved)

#label_list = sorted(set(example['label'] for example in final_dataset))
#label2id = {label: idx for idx, label in enumerate(label_list)}
#id2label = {v: k for k, v in label2id.items()}

## 라벨을 숫자로 매핑
#final_dataset = final_dataset.map(lambda x: {"label": label2id[x["label"]]})
