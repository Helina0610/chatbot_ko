# Local AI 모델 사용하기

- python : 3.12.8
- Flask : 3.1.0
- torch : 2.3.1+cu118
- transformers : 4.46.3

# AI Model
- (deepset/xlm-roberta-large-squad2)[https://huggingface.co/deepset/xlm-roberta-large-squad2]

# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/xlm-roberta-large-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/xlm-roberta-large-squad2")