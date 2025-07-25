from transformers import (
    AutoTokenizer,
    pipeline
)

PREPROCESSED_PATH = "/home/akplus/tmp_text_classifiaction/datasets/round_robin_interleaved_v1.csv"
FINE_TUNED_MODEL_PATH = "/home/akplus/tmp_text_classifiaction/fine_tuned_model_v1/model"
MODEL_PATH = "/home/akplus/smart-aide/models/multilingual-sentiment-analysis"
MAX_TOKENS = 512


## ✅ 2단계: 데이터 로딩 및 토크나이징
#dataset = load_dataset("csv", data_files=PREPROCESSED_PATH, split="train")

#label_list = sorted(set(example['label'] for example in dataset))
#label2id = {label: idx for idx, label in enumerate(label_list)}
#id2label = {v: k for k, v in label2id.items()}

## ✅ 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

#def tokenize_fn(ex):
#    return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=MAX_TOKENS)

#def encode_labels(ex):
#    ex["label"] = label2id[ex["label"]]
#    return ex

#tokenized_ds = dataset.map(tokenize_fn).map(encode_labels)

## ✅ 3단계: 모델 파인튜닝
#model = AutoModelForSequenceClassification.from_pretrained(
#    MODEL_PATH, num_labels=len(label_list), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
#)

#training_args = TrainingArguments(
#    output_dir="/home/akplus/tmp_text_classifiaction/fine_tuned_model_v1/results",
#    evaluation_strategy="no",
#    save_strategy="epoch",
#    learning_rate=2e-5,
#    per_device_train_batch_size=8,
#    num_train_epochs=3,
#    weight_decay=0.01,
#    logging_dir="/home/akplus/tmp_text_classifiaction/fine_tuned_model_v1/log",
#    save_total_limit=2,
#)

#trainer = Trainer(
#    model=model,
#    args=training_args,
#    train_dataset=tokenized_ds,
#    tokenizer=tokenizer,
#)

#trainer.train()
#trainer.save_model(FINE_TUNED_MODEL_PATH)

# ✅ 4단계: 추론 예제
pipe = pipeline("text-classification", model=FINE_TUNED_MODEL_PATH, tokenizer=tokenizer, top_k=1)

sample_text = "해당 공무원의 행동은 인사자료 통보 수준으로 판단된다."
result = pipe(sample_text)
print("예측 결과:", result[0])