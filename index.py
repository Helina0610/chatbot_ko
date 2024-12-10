from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from flask import Flask, jsonify, request

app = Flask(__name__)

model_name = "./local_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
  # 입력 데이터를 처리
    inputs = preprocess_input(question, context, tokenizer)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 모델로 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)

		# 예측된 시작과 끝 위치
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
		 # 가장 높은 확률의 시작과 끝 위치 계산
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

		 # 토크나이저를 사용해 원래 텍스트로 변환
    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1])
    return answer
  
@app.route("/", methods=['GET'])
def hello():
  return "hello"


@app.route("/chatbot", methods=['POST'])
def create_chatbot():
    # 요청 데이터 검증
    if not request.json or 'question' not in request.json or 'context' not in request.json:
        return jsonify({'error': 'The question or context is required'}), 400

    # 질문과 문맥 가져오기
    question = request.json['question']
    context = request.json['context']

    # 답변 예측
    try:
        answer = predict_answer(model, tokenizer, question=question, context=context, device=device)
        print("Predicted Answer:", answer)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
