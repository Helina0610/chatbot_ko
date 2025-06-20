import easyocr
from PIL import Image, ImageDraw, ImageFont
import os

# EasyOCR Reader 객체 생성 (한국어, 영어)
reader = easyocr.Reader(['ko', 'en'])  # 첫 실행 시 모델 다운로드됨

# 이미지 경로
img_path = 'data/9788901239156.jpg'

# OCR 수행
results = reader.readtext(img_path)

# PIL 이미지 열기
image = Image.open(img_path)
draw = ImageDraw.Draw(image)

# 한글 폰트 설정
font_path = "fonts/NanumGothic.ttf"
font = ImageFont.truetype(font_path, 18) if os.path.exists(font_path) else ImageFont.load_default()

# 결과 출력 및 시각화
for bbox, text, confidence in results:
    print(f"Text: {text}, Confidence: {confidence:.2f}")
    
    # 박스 좌표 추출
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))

    draw.rectangle([top_left, bottom_right], outline='red', width=2)
    draw.text(top_left, text, fill='red', font=font)

# 결과 이미지 저장
os.makedirs("output", exist_ok=True)
output_path = "output/easyocr_9788901239156_result.jpg"
image.save(output_path)
print(f"✅ 결과 이미지 저장 완료: {output_path}")

#Text: 너의 유토피아, Confidence: 0.62
#Text: 징보리 소설집, Confidence: 0.39
#Text: "분노하고 질문하여 멈취 애도하고, Confidence: 0.59
#Text: 2025, Confidence: 1.00
#Text: 다시 전진하는 인물들", Confidence: 0.85
#Text: 저진영(소설가), Confidence: 0.30
#Text: 필립 K 덕상, Confidence: 0.21
#Text: 최종후보작, Confidence: 0.98
#Text: 2024, Confidence: 0.55
#Text: "공포스럽고 유머러스한 이야기틀 통해, Confidence: 0.61
#Text: (타임) 선정, Confidence: 0.99
#Text: 인류의 문명올 다른다", Confidence: 0.85
#Text: {티입) 올해의 꼭 선정되, Confidence: 0.11
#Text: 올해의 책, Confidence: 0.84