from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import os

MODEL_PATH = "./models/Qwen2-0.5B-Instruct"
DATASET_PATH = "./raw_data/법령해석령(질의요지-회답)_통합.jsonl"
OUTPUT_MODEL_PATH = "./train/Qwen2-0.5B-Instruct-DPO"

# 1️⃣ 디바이스 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2️⃣ 모델 / 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 4️⃣ 데이터셋 로드
train_dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")

print(f"Sample data: {train_dataset[0]}")
print(f"model : {model.base_model_prefix}")

# 6️⃣ DPOConfig 설정
training_args = DPOConfig(
    output_dir=OUTPUT_MODEL_PATH,
    learning_rate=1e-5,
    num_train_epochs=3,
    bf16=True,
    logging_dir="tensorboard",
    save_strategy="no",
)

# 7️⃣ DPOTrainer 생성
trainer = DPOTrainer(
    model=model,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    args=training_args
)

# 8️⃣ 학습
trainer.train()

# 9️⃣ 학습 완료 모델 저장
trainer.save_model(OUTPUT_MODEL_PATH)

# 10️⃣ 저장 확인
if os.path.isdir(OUTPUT_MODEL_PATH):
    saved_files = os.listdir(OUTPUT_MODEL_PATH)
    print(f"모델이 '{OUTPUT_MODEL_PATH}'에 저장되었습니다.")
    print("저장된 파일 목록:")
    for f in saved_files:
        print(" -", f)
else:
    print(f"모델 저장 경로 '{OUTPUT_MODEL_PATH}'를 찾을 수 없습니다.")


# uv run --python 3.12 --with transformers==4.56.1 --with trl==0.23 --with datasets==4.0 --with pillow==11.3.0 --index 'https://download.pytorch.org/whl/cu13' DPO.py  
