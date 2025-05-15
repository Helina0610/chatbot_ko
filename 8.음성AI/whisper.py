import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ffmpeg 설치 경로 설정
# ffmpeg 설치 경로를 시스템 PATH에 추가
os.environ["PATH"] = r"C:\Program Files\ffmpeg\bin" + os.pathsep + os.environ["PATH"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "./model/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,  # batch size for inference - set based on your device
    torch_dtype=torch_dtype,
    device=device,
)

# 오디오 파일 추론
result = pipe("./data/chat.mp3")
print(result["text"])
