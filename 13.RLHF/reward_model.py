import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset

REWARD_MODEL_PATH = "./models/modernBERT-large-multilingual-sentiment"
POLICY_MODEL_PATH = "./models/Qwen2-0.5B-Instruct"
DATASET_PATH = "./13.RLHF/raw_data/보상모델_데이터셋.jsonl"
OUTPUT_MODEL_PATH = "./train/modernBERT-large-multilingual-sentiment-reward-model"

# 1️⃣ 디바이스 확인: GPU 사용 가능하면 CUDA, 없으면 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2️⃣ Reward 모델 로드: Sequence Classification용 pretrained 모델을 불러와서 GPU에 올림
reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH,num_labels=1).to(device)
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
max_length = 512

# 3️⃣ 데이터셋 토크나이징 함수 (pairwise)
def tokenize_pairwise(example):
    good_input = reward_tokenizer(
        example["prompt"],
        example["chosen"],
        truncation=True,
        max_length=max_length, 
        padding="max_length"
    )
    bad_input = reward_tokenizer(
        example["prompt"],
        example["rejected"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
     # pairwise 데이터를 RewardTrainer가 이해할 수 있는 형태로 반환
    return {
        "input_ids_chosen": good_input["input_ids"],
        "attention_mask_chosen": good_input["attention_mask"],
        "input_ids_rejected": bad_input["input_ids"],
        "attention_mask_rejected": bad_input["attention_mask"]
    }

# 4️⃣ JSONL 데이터셋 로드
raw_datasets = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")

# 5️⃣ 데이터셋 토크나이징 적용
tokenized_datasets = raw_datasets.map(tokenize_pairwise, batched=False)

# 6️⃣ RewardTrainer 학습 설정
reward_config = RewardConfig(
    dataset_num_proc=1,
    center_rewards_coefficient=0.01,
    remove_unused_columns=False,
    output_dir=OUTPUT_MODEL_PATH,
    save_strategy="no",
)

# 7️⃣ RewardTrainer 초기화
trainer = RewardTrainer(
    model=reward_model,
    args=reward_config,
    train_dataset=tokenized_datasets,
    processing_class=reward_tokenizer,
)

# 8️⃣ Reward 모델 학습
trainer.train()

# 9️⃣ 학습 완료 후 모델 저장
trainer.save_model(OUTPUT_MODEL_PATH)


# 학습 완료 모델 저장 확인
if os.path.isdir(OUTPUT_MODEL_PATH):
    saved_files = os.listdir(OUTPUT_MODEL_PATH)
    print(f"모델이 '{OUTPUT_MODEL_PATH}'에 저장되었습니다.")
    print("저장된 파일 목록:")
    for f in saved_files:
        print(" -", f)
else:
    print(f"모델 저장 경로 '{OUTPUT_MODEL_PATH}'를 찾을 수 없습니다.")
    
# uv로 실행
# uv run --python 3.12 --with transformers==4.56.1 --with trl==0.23 --with datasets==4.0 --index 'https://download.pytorch.org/whl/cu13' reward_model.py  
    
    
#모델이 './train/modernBERT-large-multilingual-sentiment-reward-model'에 저장되었습니다.
#저장된 파일 목록:
# - config.json
# - model.safetensors
# - tokenizer_config.json
# - special_tokens_map.json
# - tokenizer.json
# - training_args.bin