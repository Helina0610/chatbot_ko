from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import requests 
import unicodedata
from io import BytesIO
from PIL import Image

processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr") 
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

url = "https://contents.kyobobook.co.kr/sih/fit-in/458x0/pdt/9791168342569.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

pixel_values = processor(img, return_tensors="pt").pixel_values 
generated_ids = model.generate(pixel_values, max_length=64)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
generated_text = unicodedata.normalize("NFC", generated_text)
print(generated_text)