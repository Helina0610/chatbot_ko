import json
import os
import re
from uuid import uuid4

from datasets import load_dataset
from kiwipiepy import Kiwi
from langchain_community.document_loaders import PDFPlumberLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

# ✅ 설정
MODEL_PATH = "local_model/multilingual-sentiment-analysis"
PDF_DIR = "E:/hj_code/유사사례데이터/"  # 라벨은 data/고발, data/모범, data/변상
MAX_TOKENS = 512
STRIDE = 256
MIN_TOKENS = 10
PREPROCESSED_PATH = "text_classification/data/preprocessed_data_test1.jsonl"

# ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ 문장 기반 청크 생성
#kiwi = Kiwi()
#def sentence_based_chunk(text, max_tokens=512, stride=256):
#    try:
#        split_result = kiwi.split_into_sents(text)
#        sentences = [s.text for s in split_result]
#    except Exception as e:
#        print(f"문장 분리 실패: {e}")
#        sentences = [text]

#    chunks, current_chunk = [], []
#    current_length = 0

#    for sent in sentences:
#        token_length = len(tokenizer.tokenize(sent))
#        if token_length > max_tokens:
#            continue
#        if current_length + token_length > max_tokens:
#            chunks.append(" ".join(current_chunk))
#            current_chunk = [sent]
#            current_length = token_length
#        else:
#            current_chunk.append(sent)
#            current_length += token_length

#    if current_chunk:
#        chunks.append(" ".join(current_chunk))

#    return chunks

## 텍스트 전처리
#def normalize_text(text) -> str:
#    # 불필요한 공백 제거
#    text = re.sub(r"\s+", " ", text)
#    # 소문자
#    text = text.lower()
#     # 여러 공백 → 하나
#    text = re.sub(r"\s+", " ", text) 
#    return text

## ✅ 1단계: PDF 전처리 및 저장

#for root, dirs, files in os.walk(PDF_DIR):
#    # 경로 안에 라벨 이름이 포함되어 있는지 확인
#    matched_label = None
#    for label_name in ["고발"]:
#        if label_name in root:
#            matched_label = label_name
#            break

#    if matched_label is None:
#        continue

#    print(f"▶ 라벨: {matched_label}, 폴더: {root}")
#    PREPROCESSED_PATH = f"text_classification/datasets/preprocessed_{matched_label}.jsonl"
#    os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)

#    for file in files:
#        examples = []
#        if not file.endswith(".pdf"):
#            continue

#        pdf_path = os.path.join(root, file)
#        try:
#            loader = PDFPlumberLoader(file_path=pdf_path)
#            documents = loader.load()
#            full_text = " ".join([doc.page_content for doc in documents])
#            parser_text = normalize_text(full_text)
#            chunks = sentence_based_chunk(parser_text)
#            print(len(chunks))

#            for chunk in chunks:
#                tokens = tokenizer.encode(chunk, truncation=True, max_length=MAX_TOKENS)
#                if len(tokens) >= MIN_TOKENS:
#                    examples.append({"label": matched_label, "text": chunk})

#            with open(PREPROCESSED_PATH, "a", encoding="utf-8") as f:
#                for ex in examples:
#                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
#            print(f"✅ PDF 처리 성공: {pdf_path} - {len(chunks)} 청크")

#        except Exception as e:
#            print(f"❌ PDF 처리 실패: {pdf_path} - {e}")

## ✅ JSONL 저장
#os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)
#with open(PREPROCESSED_PATH, "w", encoding="utf-8") as f:
#    for ex in examples:
#        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

PREPROCESSED_PATH = "text_classification/data/preprocessed_data_test1.jsonl"
FINE_TUNED_MODEL_PATH = "fine_tuned_model"

# ✅ 2단계: 데이터 로딩 및 토크나이징
dataset = load_dataset("json", data_files=PREPROCESSED_PATH, split="train")
label_list = sorted(set(example['label'] for example in dataset))
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}

def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=MAX_TOKENS)

def encode_labels(ex):
    ex["label"] = label2id[ex["label"]]
    return ex

tokenized_ds = dataset.map(tokenize_fn).map(encode_labels)

# ✅ 3단계: 모델 파인튜닝
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=len(label_list), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(FINE_TUNED_MODEL_PATH)

# ✅ 4단계: 추론 예제
pipe = pipeline("text-classification", model=FINE_TUNED_MODEL_PATH, tokenizer=tokenizer, top_k=1)

sample_text = "해당 공무원의 행동은 인사자료 통보 수준으로 판단된다."
result = pipe(sample_text)
print("예측 결과:", result[0])


#import pandas as pd
#import pdfplumber


#with pdfplumber.open("data/터널 바닥면 고르기 공사비 산정 부적정15124.pdf") as pdf:
#    pages = pdf.pages
#    print(f"총 페이지 수: {len(pages)}")
#    for page in pages:
#        table = page.extract_table()
#        print(f"페이지 {page.page_number}의 테이블:")
    
#        #if table:
#        #    df = pd.DataFrame(table[1:], columns=table[0])
#        #    markdown_table = df.to_markdown(index=False)
#        #    print(markdown_table)