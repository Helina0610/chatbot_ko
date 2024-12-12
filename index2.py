import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from flask import Flask, jsonify, request
import pymupdf 

app = Flask(__name__)

model_name = "./local_model"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 문맥 나누기 함수: 겹치는 청크 생성
def split_with_overlap(text, max_length=512, overlap=128):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i+max_length])
        chunks.append(chunk)
        if i + max_length >= len(words):
            break
    return chunks

# 긴 문맥에 대한 전처리 함수
def preprocess_long_context(question, context, tokenizer, max_length=512, overlap=128):
    chunks = split_with_overlap(context, max_length=max_length, overlap=overlap)
    inputs = []
    
    for chunk in chunks:
        input_data = tokenizer(
            question,
            chunk,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        inputs.append(input_data)
    return inputs
  
# 긴 문맥에 대해 모델 응답 생성 및 병합
def generate_answer(question, context, tokenizer, model, max_length=512, overlap=128):
    inputs = preprocess_long_context(question, context, tokenizer, max_length, overlap)
    answers = []

    for input_data in inputs:
        with torch.no_grad():
            outputs = model(**input_data)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # 가장 높은 점수를 가진 start와 end 위치 선택
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
            
            # 정답 추출
            answer = tokenizer.decode(
                input_data["input_ids"][0][start_idx:end_idx + 1],
                skip_special_tokens=True
            )
            answers.append(answer)
    
    # 모든 청크에서 나온 응답 병합
    final_answer = " ".join(answers)
    return final_answer

# pdf 텍스트 처리
def extract_text_from_pdf_fitz(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
      page = doc.load_page(page_num)
      text += page.get_text("text")
    return text

  
@app.route("/", methods=['GET'])
def hello():
  return jsonify({'answer': "hello!"})

@app.route("/chatbot", methods=['POST'])
def create_chatbot():
    # 요청 데이터 검증
    if not request.json or 'qustCont' not in request.json or 'url' not in request.json:
        return jsonify({'error': 'The question or context is required'}), 400

    # 질문과 문맥 가져오기
    question = request.json['qustCont']
    pdf_path = request.json['url']
    #pdf_path = 'E:/hj_code/Python/chatbot_ko_v2/data/sampleAll.pdf'
    print(pdf_path)
    
    context = extract_text_from_pdf_fitz(pdf_path)
    single_line_text = " ".join(context.split())

    # 답변 예측
    try:
        answer =  answer = generate_answer(question, context=single_line_text, tokenizer=tokenizer, model=model)
        print("Predicted Answer:", answer)
        return jsonify({'answer': answer}),200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)