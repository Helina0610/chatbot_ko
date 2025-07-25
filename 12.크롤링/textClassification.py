from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load
import pandas as pd
import torch
import numpy as np

# 1. 데이터 로딩
dataset_csv = load_dataset("csv", data_files={"train": "text_classification/output/selenium_crawled_data.csv"})

# 2. 라벨 매핑
label_list = dataset_csv["train"].unique("label")
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {v: k for k, v in label2id.items()}

#def encode_labels(example):
#    example["labels"] = label2id[example["label"]]
#    return example

#dataset = dataset_csv["train"].map(encode_labels)

## 3. train/test 분리
#dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 4. 토크나이저 로딩
model_name = "local_model/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#def tokenize(example):
#    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

#tokenized_dataset = dataset.map(tokenize, batched=True)
#tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

## 5. 모델 로딩
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

## 6. 훈련 설정
#training_args = TrainingArguments(
#    output_dir="text_classification/result",
#    evaluation_strategy="epoch",
#    save_strategy="epoch",
#    logging_dir="text_classification/logs",
#    num_train_epochs=3,
#    per_device_train_batch_size=8,
#    per_device_eval_batch_size=8,
#    load_best_model_at_end=True,
#    metric_for_best_model="accuracy"
#)

## 7. 메트릭 함수
#accuracy = load("accuracy")

#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return accuracy.compute(predictions=predictions, references=labels)

## 8. Trainer
#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=tokenized_dataset["train"],
#    eval_dataset=tokenized_dataset["test"],
#    compute_metrics=compute_metrics,
#    tokenizer=tokenizer,
#)

#trainer.train()

# 9. 예측 함수
def predict_verbose(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    for i, p in enumerate(probs):
        print(f"{id2label[i]}: {p.item():.4f}")
    pred_id = torch.argmax(probs).item()
    return id2label[pred_id]

prompt = "한국산업은행 회장에게 신용등금을 상향 조정할 때 증빙자료를 첨부하지 않거나 상향 사유가 실현되지 않았는데도 신용등급을 그대로 유지하는 일이 없게 사후관리 등 관련 업무를 철저히 하도록 주의"
print(predict_verbose(prompt))


 