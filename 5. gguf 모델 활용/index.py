from llama_cpp import Llama
from transformers import AutoTokenizer
from flask import Flask, jsonify, request
from werkzeug.exceptions import InternalServerError

app = Flask(__name__)

# model, tokenizer 불러오기
tokenizer_path = './local_model/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/tokenizer'
model_path = './local_model/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf'

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
except Exception as e:
    raise InternalServerError(f"토크나이저 로드 중 오류가 발생했습니다: {str(e)}")

try:
    model = Llama(model_path=model_path)
except Exception as e:
    raise InternalServerError(f"모델 로드 중 오류가 발생했습니다: {str(e)}")

def success_parsed_text(response_msg):
    # 응답 메시지에서 텍스트 추출
    text = response_msg['choices'][0]['text']
    answer = text.split("<|start_header_id|>assistant<|end_header_id|>")[1]

    obj = {
        'answer': answer,
        'references': [
            {
                'fileName': '',
                'title': '',
                'page': 1,
                'text': ''
            }
        ],
        'created': response_msg['created'],
        'model': response_msg['model'],
        'status': 200
    }
    return obj

def handle_error(error,status=500):

    return jsonify({
      "errorMessage": str(error),
      "errorCode": "CHAT_ERROR",
      "status": status
    }), 500

@app.route("/", methods=['GET'])
def hello():
    return jsonify({'answer': "hello!"})

@app.route("/chatbot", methods=['POST'])
def create_chatbot():
    if not request.args.get("message", ""):
        return handle_error("The question is required", 400)

    instruction = request.args.get("message", "")

    messages = [
        {"role": "user", "content": f"{instruction}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    generation_kwargs = {
        "max_tokens": 512,
        "stop": ["<|eot_id|>"],
        "echo": True,
        "top_p": 0.9,
        "temperature": 0.6,
    }

    try:
        response_msg = model(prompt, **generation_kwargs)
        parsed_data = success_parsed_text(response_msg)
        return jsonify(parsed_data), 200
    except Exception as e:
        return handle_error(f"응답 생성 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
