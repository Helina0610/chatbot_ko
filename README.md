# Local AI 모델 사용하기

- python : 3.12.8
- Flask : 3.1.0
- torch : 2.3.1+cu118
- transformers : 4.46.3

# AI Model
- [deepset/xlm-roberta-large-squad2](https://huggingface.co/deepset/xlm-roberta-large-squad2)

# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/xlm-roberta-large-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/xlm-roberta-large-squad2")

# Guide
미래의 나를 위해 작성된 가이드
- [Flask 설치](Guide/Flask_설치.md)
- [Local Al Model 사용하기](Guide/Local_AI_%20Model_사용하기.md)
