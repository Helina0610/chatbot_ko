from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# 모델과 토크나이저 로드
model_name = "./local_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_name,ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 질문과 문맥을 토근화
def preprocess_input(question, context, tokenizer, max_length=512):
    inputs = tokenizer(
        question, 
        context, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors="pt"
    )
    return inputs

def predict_answer(model, tokenizer, question, context, device):
    inputs = preprocess_input(question, context, tokenizer)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1])
    return answer

# 예제 입력
#question = "Why is model conversion important?"
#context = (
#    "The option to convert models between FARM and transformers gives freedom "
#    "to the user and let people easily switch between frameworks."
#)


context = ("한국의 전통 문화는 오랜 역사와 깊은 의미를 담고 있습니다. 한국의 대표적인 전통 예술로는 한지(한국 전통 종이), 판소리(전통 음악과 이야기), 한복(전통 의상) 등이 있습니다. 한지는 천연 재료로 만들어져 뛰어난 내구성을 자랑하며, 다양한 전통 공예에 사용됩니다. 판소리는 한 명의 가수가 서사적인 이야기를 노래로 풀어내는 한국의 전통 음악으로, 유네스코 무형문화유산으로 등재된 바 있습니다. 또한, 한복은 그 자체로 한국의 아름다움과 정체성을 상징하는 의상으로, 특별한 날에 착용됩니다.")
question = "판소리는?"

print(len(context))
# 답변 예측
#answer = predict_answer(model, tokenizer, question, context, device)
#print("Predicted Answer:", answer)
