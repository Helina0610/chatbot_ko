# 1. AI 모델 다운로드
- Huggingface 에서 다운로드 받을 모델의 페이지로 이동합니다.
## 1.1 `Git` 을 이용하여 다운로드 받기

- 대용량 파일을 받을 때, `git lfs` 를 사용해서 받을 수 있다

```powershell
git isf install
```

- 다운로드 받을 모델의 Huggingface url ,다운로드 받을 경로를 뒤에 써서 clone 한다

```powershell
# git clone [HuggingFace URL] [다운로드 받을 경로]

git clone <https://huggingface.co/deepset/xlm-roberta-large-squad2> E:/hj_code/Python/flask/local_model
```

## 1.2. Files and Versions 다운로드 받기
- Files and versions 에서 직접 다운로드 받기

# 2. 관련 패키지 설치하기
```powershell
pip install transformers torch Flask tokenizers protobuf sentencepiece
```

# 3. 로컬 AI Model 사용하기
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "./local_model"
tokenizer = AutoTokenizer.from_pretrained(model_name )
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
```
- 다운로드 받은 AI Model 의 경로를 model_name 으로 정해서 토큰화, 모델 만들기