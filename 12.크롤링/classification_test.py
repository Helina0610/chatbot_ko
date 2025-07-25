# 설치 필요 시
# pip install sentence-transformers scikit-learn

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


# 1. 문서 분할 함수
def split_document(text: str, max_tokens: int = 200) -> list[str]:
    sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_length = len(sentence.split())
        if current_length + token_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = token_length
        else:
            current_chunk.append(sentence)
            current_length += token_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# 2. 문서 임베딩 함수 (평균 임베딩 방식)
def get_document_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    chunks = split_document(text)
    if not chunks:
        return np.zeros(model.get_sentence_embedding_dimension())

    embeddings = model.encode(chunks)
    return np.mean(embeddings, axis=0)


# 3. 예시 데이터 (실제 데이터로 교체하세요)
documents = [
    "이 문서는 고양이에 대한 설명으로 구성되어 있습니다. 고양이는 포유류입니다. 야행성이며 독립적인 성격을 가집니다.",
    "이 문서는 강아지에 관한 내용입니다. 강아지는 사람과 친숙하며 충성심이 강합니다. 애완동물로 많이 길러집니다.",
    "강아지와 고양이는 인기 있는 반려동물입니다. 각각의 특징이 다르며, 사람들의 취향에 따라 선택됩니다.",
    "고양이는 깨끗한 동물로 알려져 있습니다. 스스로 그루밍을 하며, 조용한 성격을 가졌습니다.",
    "강아지는 산책을 좋아하며 주인에게 애정을 자주 표현합니다. 교육이 상대적으로 용이합니다."
]
labels = [0, 1, 1, 0, 1]  # 0: 고양이, 1: 강아지 (예시)

# 4. 모델 로딩
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5. 문서 임베딩 추출
X = [get_document_embedding(doc, model) for doc in documents]
y = labels

# 6. 분류기 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

# 7. 평가
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["고양이", "강아지"]))

#import os
#from glob import glob
#from langchain_community.document_loaders import PyPDFLoader
#import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
#from sentence_transformers import SentenceTransformer

## 1. 문서 분할 함수
#def split_document(text: str, max_tokens: int = 200) -> list[str]:

#    sentences = text.split(". ")
#    chunks = []
#    current_chunk = []
#    current_length = 0

#    for sentence in sentences:
#        token_length = len(sentence.split())
#        if current_length + token_length > max_tokens:
#            chunks.append(" ".join(current_chunk))
#            current_chunk = [sentence]
#            current_length = token_length
#        else:
#            current_chunk.append(sentence)
#            current_length += token_length

#    if current_chunk:
#        chunks.append(" ".join(current_chunk))

#    return chunks
  
## 2. 문서 임베딩 함수 (평균 임베딩 방식)
#def get_document_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
#    chunks = split_document(text)
#    if not chunks:
#        return np.zeros(model.get_sentence_embedding_dimension())

#    embeddings = model.encode(chunks)
#    return np.mean(embeddings, axis=0)


#LOCAL_MODEL_PATH = "local_model/multilingual-e5-small-ko"
#model = SentenceTransformer(LOCAL_MODEL_PATH)

## 파일 목록
#labels = ["인사자료통보","주의","징계문책","통보권고"]
#docs = []
#pdf_files = glob(os.path.join('data/data', '*처분요구별 공개문-*.pdf'))
#print(len(pdf_files))
#for pdf in pdf_files:
#    loader = PyPDFLoader(file_path=pdf, mode="single")
#    documents = loader.load()
#    for document in documents:  # documents 내부 Document 하나씩 꺼내기
#        docs.append(document)

#print(len(docs))
#for doc in docs:
#    X = [get_document_embedding(doc.page_content, model) for doc in docs]
#    Y = labels

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#clf = LogisticRegression()
#clf.fit(X_train, y_train)

## 7. 평가
#y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred, target_names=["통보권고", "주의"]))


#import os
#from glob import glob
#from langchain_community.document_loaders import PyPDFLoader
#import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
#from sentence_transformers import SentenceTransformer


## 1. 문서 분할 함수
#def split_document(text: str, max_tokens: int = 200) -> list[str]:
#    sentences = text.split(". ")
#    chunks = []
#    current_chunk = []
#    current_length = 0

#    for sentence in sentences:
#        token_length = len(sentence.split())
#        if current_length + token_length > max_tokens:
#            chunks.append(" ".join(current_chunk))
#            current_chunk = [sentence]
#            current_length = token_length
#        else:
#            current_chunk.append(sentence)
#            current_length += token_length

#    if current_chunk:
#        chunks.append(" ".join(current_chunk))
#    return chunks


## 2. 문서 임베딩 함수 (평균 임베딩 방식)
#def get_document_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
#    chunks = split_document(text)
#    if not chunks:
#        return np.zeros(model.get_sentence_embedding_dimension())
#    embeddings = model.encode(chunks)
#    return np.mean(embeddings, axis=0)


## 3. 모델 로딩
#LOCAL_MODEL_PATH = "local_model/multilingual-e5-small-ko"
#model = SentenceTransformer(LOCAL_MODEL_PATH)


## 4. 문서 임베딩 및 라벨 추출
#X = []
#Y = []

#pdf_files = glob(os.path.join('data/data', '*처분요구별 공개문-*.pdf'))
#print(f"PDF 파일 수: {len(pdf_files)}")

#for pdf in pdf_files:
#    loader = PyPDFLoader(file_path=pdf, mode="single")
#    documents = loader.load()

#    # 전체 페이지 텍스트 이어붙이기
#    full_text = " ".join([doc.page_content for doc in documents])
#    embedding = get_document_embedding(full_text, model)
#    if np.count_nonzero(embedding) == 0:
#        continue

#    X.append(embedding)

#    # 파일명에서 라벨 추출
#    filename = os.path.basename(pdf)
#    if "인사자료통보" in filename:
#        Y.append("인사자료통보")
#    elif "주의" in filename:
#        Y.append("주의")
#    elif "징계문책" in filename:
#        Y.append("징계문책")
#    elif "통보권고" in filename:
#        Y.append("통보권고")
#    else:
#        print(f"⚠️ 라벨을 추출할 수 없음: {filename}")


#print(f"총 문서 수: {len(X)} / 라벨 수: {len(Y)}")

#if len(X) < 2:
#    print("❌ 학습할 데이터가 부족합니다.")
#    exit()

## 5. 분류기 학습
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#clf = LogisticRegression(max_iter=1000)
#clf.fit(X_train, y_train)

#import joblib

## 학습된 모델 저장
#joblib.dump(clf, "classifier.pkl")
#print("✅ LogisticRegression 모델이 classifier.pkl 파일로 저장되었습니다.")

#import joblib

## 모델 불러오기
#clf = joblib.load("classifier.pkl")
#print("✅ 저장된 분류기 모델을 불러왔습니다.")

## 예측 가능
#y_pred = clf.predict([new_embedding])



## 6. 평가
#y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred))



#def classify_new_document(pdf_path: str) -> str:
#    loader = PyPDFLoader(file_path=pdf_path, mode="single")
#    documents = loader.load()
#    full_text = " ".join([doc.page_content for doc in documents])
#    embedding = get_document_embedding(full_text, model)

#    if np.count_nonzero(embedding) == 0:
#        return "❌ 텍스트를 추출할 수 없습니다."

#    prediction = clf.predict([embedding])[0]
#    return prediction


#new_pdf = "data/test/처분요구별 공개문-새로운문서.pdf"
#predicted_label = classify_new_document(new_pdf)
#print(f"예측된 라벨: {predicted_label}")

#import os
#from glob import glob
#import joblib
#import numpy as np
#from langchain_community.document_loaders import PyPDFLoader
#from sentence_transformers import SentenceTransformer
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report

## 문서 분할 함수
#def split_document(text: str, max_tokens: int = 200) -> list[str]:
#    sentences = text.split(". ")
#    chunks, current_chunk, current_length = [], [], 0
#    for sentence in sentences:
#        token_length = len(sentence.split())
#        if current_length + token_length > max_tokens:
#            chunks.append(" ".join(current_chunk))
#            current_chunk = [sentence]
#            current_length = token_length
#        else:
#            current_chunk.append(sentence)
#            current_length += token_length
#    if current_chunk:
#        chunks.append(" ".join(current_chunk))
#    return chunks

## 임베딩 함수
#def get_document_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
#    chunks = split_document(text)
#    if not chunks:
#        return np.zeros(model.get_sentence_embedding_dimension())
#    embeddings = model.encode(chunks)
#    return np.mean(embeddings, axis=0)

## PDF → 임베딩 + 라벨 추출
#def build_dataset(pdf_files: list[str], model: SentenceTransformer):
#    X, Y = [], []
#    for pdf in pdf_files:
#        loader = PyPDFLoader(file_path=pdf, mode="single")
#        documents = loader.load()
#        full_text = " ".join([doc.page_content for doc in documents])
#        embedding = get_document_embedding(full_text, model)
#        if np.count_nonzero(embedding) == 0:
#            continue
#        X.append(embedding)

#        # 라벨 추출
#        filename = os.path.basename(pdf)
#        if "인사자료통보" in filename:
#            Y.append("인사자료통보")
#        elif "주의" in filename:
#            Y.append("주의")
#        elif "징계문책" in filename:
#            Y.append("징계문책")
#        elif "통보권고" in filename:
#            Y.append("통보권고")
#        else:
#            print(f"[경고] 라벨 인식 실패: {filename}")
#    return X, Y

## 문서 분류 함수 (새 문서)
#def classify_new_document(pdf_path: str, model: SentenceTransformer, clf, label_list):
#    loader = PyPDFLoader(file_path=pdf_path, mode="single")
#    documents = loader.load()
#    full_text = " ".join([doc.page_content for doc in documents])
#    embedding = get_document_embedding(full_text, model)
#    if np.count_nonzero(embedding) == 0:
#        return "❌ 텍스트를 추출할 수 없습니다."
#    pred = clf.predict([embedding])[0]
#    return pred

## ---------- 메인 파이프라인 ----------
#if __name__ == "__main__":
#    LOCAL_MODEL_PATH = "local_model/multilingual-e5-small-ko"
#    EMBEDDING_MODEL = SentenceTransformer(LOCAL_MODEL_PATH)

#    # Step 1: 학습용 PDF 읽기
#    pdf_files = glob(os.path.join("data/data", "*처분요구별 공개문-*.pdf"))
#    print(f"학습용 PDF 수: {len(pdf_files)}")

#    # Step 2: 임베딩 + 라벨 추출
#    X, Y = build_dataset(pdf_files, EMBEDDING_MODEL)
#    print(f"임베딩된 문서 수: {len(X)}, 라벨 수: {len(Y)}")

#    # Step 3: 분류기 학습
#    if len(X) < 2:
#        print("❌ 학습 데이터가 부족합니다.")
#        exit()

#    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#    clf = LogisticRegression(max_iter=1000)
#    clf.fit(X_train, y_train)

#    # Step 4: 평가
#    y_pred = clf.predict(X_test)
#    print("\n📊 분류 성능:")
#    print(classification_report(y_test, y_pred))

#    # Step 5: 모델 저장
#    joblib.dump(clf, "classifier.pkl")
#    joblib.dump(clf.classes_, "labels.pkl")
#    print("\n✅ 모델과 라벨을 저장했습니다.")

#    # Step 6: 테스트 문서 분류 예시
#    test_pdf = "data/test/처분요구별 공개문-테스트.pdf"  # 예시 파일 경로
#    if os.path.exists(test_pdf):
#        clf_loaded = joblib.load("classifier.pkl")
#        label_list = joblib.load("labels.pkl")
#        result = classify_new_document(test_pdf, EMBEDDING_MODEL, clf_loaded, label_list)
#        print(f"\n📝 테스트 문서 예측 결과: {result}")
#    else:
#        print("⚠️ 테스트 문서가 존재하지 않습니다.")

#import os
#from glob import glob
#from tqdm import tqdm
#import json
#import nltk
#import torch
#from transformers import AutoTokenizer
#from langchain_community.document_loaders import PyPDFLoader

#nltk.download('punkt')

## ✅ 설정
#MODEL_PATH = "local_model/multilingual-sentiment-analysis"
#PDF_DIR = "data/data"
#MAX_TOKENS = 512
#STRIDE = 256
#MIN_TOKENS = 10

## ✅ 토크나이저 로드
#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

## ✅ 문장 기반 청크 생성 함수
#def sentence_based_chunk(text, max_tokens=MAX_TOKENS, stride=STRIDE):
#    sentences = nltk.sent_tokenize(text)
#    chunks, current_chunk = [], []
#    current_length = 0

#    for sent in sentences:
#        token_length = len(tokenizer.tokenize(sent))
#        if token_length > max_tokens:
#            continue  # 너무 긴 문장은 버림
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

## ✅ 라벨 추출 함수 (파일명 기반)
#def extract_label_from_filename(filename):
#    if "인사자료통보" in filename:
#        return "인사자료통보"
#    elif "주의" in filename:
#        return "주의"
#    elif "징계문책" in filename:
#        return "징계문책"
#    elif "통보권고" in filename:
#        return "통보권고"
#    return None

## ✅ PDF 문서 처리 및 청크 전처리
#examples = []
#pdf_files = glob(os.path.join(PDF_DIR, "*처분요구별 공개문-*.pdf"))

#for pdf in tqdm(pdf_files):
#    label = extract_label_from_filename(pdf)
#    if not label:
#        continue

#    loader = PyPDFLoader(file_path=pdf, mode="single")
#    documents = loader.load()
#    full_text = " ".join([doc.page_content for doc in documents])

#    chunks = sentence_based_chunk(full_text)
#    for chunk in chunks:
#        tokens = tokenizer.encode(chunk, truncation=True, max_length=MAX_TOKENS)
#        if len(tokens) >= MIN_TOKENS:
#            examples.append({
#                "text": chunk,
#                "label": label
#            })

## ✅ 결과 저장
#data_path = "preprocessed_data.jsonl"
#with open(data_path, "w", encoding="utf-8") as f:
#    for ex in examples:
#        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

#print(f"✅ 총 {len(examples)}건 청크 저장 완료: {data_path}")

##✅ 1. HuggingFace datasets로 불러오기
#from datasets import load_dataset

#dataset = load_dataset("json", data_files="preprocessed_data.jsonl", split="train")
#label_list = sorted(set(example['label'] for example in dataset))
#label2id = {label: idx for idx, label in enumerate(label_list)}
#id2label = {v: k for k, v in label2id.items()}

#def tokenize_fn(ex):
#    return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512)

#def encode_labels(ex):
#    ex["label"] = label2id[ex["label"]]
#    return ex

#tokenized_ds = dataset.map(tokenize_fn).map(encode_labels)

## ✅ 2. Trainer로 파인튜닝
#from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#model = AutoModelForSequenceClassification.from_pretrained(
#    MODEL_PATH, num_labels=len(label_list), id2label=id2label, label2id=label2id
#)

#training_args = TrainingArguments(
#    output_dir="./results",
#    evaluation_strategy="no",
#    save_strategy="epoch",
#    learning_rate=2e-5,
#    per_device_train_batch_size=8,
#    num_train_epochs=3,
#    weight_decay=0.01,
#    logging_dir="./logs",
#    save_total_limit=2,
#)

#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=tokenized_ds,
#    tokenizer=tokenizer,
#)

#trainer.train()
#trainer.save_model("fine_tuned_model")

## ✅ 3. 추론
#from transformers import pipeline

#pipe = pipeline("text-classification", model="fine_tuned_model", tokenizer=tokenizer, top_k=1)

#text = "공무원의 태도는 징계문책 사유가 됩니다."
#pred = pipe(text)
#print(pred[0])
