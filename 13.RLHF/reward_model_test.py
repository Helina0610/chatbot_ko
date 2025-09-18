import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 저장된 reward 모델 경로
MODEL_PATH = "./train/modernBERT-large-multilingual-sentiment-reward-model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 & 토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 테스트 프롬프트와 답변
data = {
    "prompt": "사립 고등학교의 위치를 변경하기 위해 「사립학교법」 제29조제2항에 따른 교비회계에 속하는 기존 학교의 교육용 기본재산을 처분하고 그 처분대금으로 새로운 학교의 교육용 기본재산을 확보하는 경우, 새로운 학교의 교육용 기본재산을 확보하고 남은 처분대금은 교비회계에 보전해야 하는지?",
    "chosen": "이 사안의 경우 새로운 학교의 교육용 기본재산을 확보하고 남은 처분대금은 교비회계에 보전해야 합니다.",
    "rejected": "학교의 새로운 교육재산을 구매하지 말고, 기존 교비에서 먼저 사용하세요."
}
prompt = data["prompt"]
chosen = data["chosen"]
rejected  = data["rejected"]

def get_reward(prompt, response):
    inputs = tokenizer(
        prompt, response,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch_size, num_labels)

        # ✅ 보상 점수를 스칼라로 변환
        if logits.shape[-1] == 1:
            reward = logits.squeeze().item()  # (batch_size, 1) → scalar
        else:
            # num_labels > 1인 경우 → 첫 번째 로짓을 보상 점수로 사용 (혹은 softmax 차이값)
            reward = logits[:, 0].item()

    return reward

print("Chosen Reward:", get_reward(prompt, chosen))
print("Rejected Reward:", get_reward(prompt, rejected))

# uv run --python 3.12 --with transformers --with torch --index 'https://download.pytorch.org/whl/cu13' test_reward_model.py  
