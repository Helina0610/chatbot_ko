import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json

# 1. Hugging Face 모델 로드
model_name = "./local_model/llama-3.2-Korean-Bllossom-AICA-5B"  # 모델 이름 또는 경로
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 2. GGUF 구조 정의 및 저장
def convert_to_gguf(model, tokenizer, output_path):
    # 모델의 가중치와 tokenizer 데이터를 추출
    gguf_data = {
        "vocab": tokenizer.get_vocab(),  # 토크나이저의 vocab을 가져옵니다.
        "config": model.config.to_dict(),  # 모델의 설정 정보
        "weights": {name: param.detach().cpu().numpy().tolist() for name, param in model.named_parameters()},
    }

    # GGUF 파일로 저장 (구조에 맞게)
    with open(output_path, "wb") as f:
        # GGUF 헤더 작성 (여기서는 예시로 "gguf_format"을 사용)
        header = b"gguf_format_v1"  # 버전 정보 추가 가능
        f.write(header)

        # 모델 메타데이터 저장 (JSON 형식으로 저장 가능)
        json_data = json.dumps(gguf_data).encode('utf-8')
        f.write(len(json_data).to_bytes(4, byteorder='little'))  # 데이터 길이 (4바이트)
        f.write(json_data)

        # 모델 가중치 저장 (각 파라미터 데이터를 순차적으로 저장)
        for name, weight in gguf_data['weights'].items():
            weight_data = np.array(weight).tobytes()  # 리스트를 numpy 배열로 변환 후 바이트로 변환
            f.write(len(weight_data).to_bytes(4, byteorder='little'))  # 데이터 길이 (4바이트)
            f.write(weight_data)

output_file = "llama-3.2-Korean-Bllossom-AICA-5B.gguf"
convert_to_gguf(model, tokenizer, output_file)
print(f"GGUF 파일로 변환 완료: {output_file}")

