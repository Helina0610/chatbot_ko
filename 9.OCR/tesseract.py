import pytesseract
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

# Tesseract 실행 파일 경로 지정 (Windows 사용자만 필요)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 이미지 경로
img_path = 'data/9788901239156.jpg'


# PIL 이미지로 변환 (텍스트 표시용)
image = Image.open(img_path).convert('RGB')
draw = ImageDraw.Draw(image)

# 한글 폰트 지정 (경로에 따라 수정)
font_path = "fonts/NanumGothic.ttf"
if not os.path.exists(font_path):
    font = ImageFont.load_default()
else:
    font = ImageFont.truetype(font_path, 16)

# OCR with 위치 정보
data = pytesseract.image_to_data(image, lang='kor+eng', output_type=pytesseract.Output.DICT)
data2 = pytesseract.image_to_string(image, lang='kor+eng', output_type=pytesseract.Output.STRING)
print(data2)
#“분노하고 질문하며 멈춰 애도하고

#다시 전진하는 인물들”

#aaa

#“공포스럽고 유머러스한 이야기를 통해

#인류의 BBS 다룬다"

#2025
#필립 <. 딕상
#최종후보작

#2024
#(타임) 선정
#올해의책


# 박스와 텍스트 그리기
n_boxes = len(data['level'])
for i in range(n_boxes):
    text = data['text'][i]
    print(f"Text: {text}, Confidence: {data['conf'][i]}")
    conf = int(data['conf'][i])

    if conf > 60 and text.strip():  # 신뢰도와 공백 필터링
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        draw.rectangle([(x, y), (x + w, y + h)], outline='red', width=2)
        draw.text((x, y - 18), text, fill='red', font=font)

# 결과 저장
os.makedirs("output", exist_ok=True)
output_path = "output/tesseract_9788901239156_result.jpg"
image.save(output_path)
print(f"✅ 결과 저장 완료: {output_path}")