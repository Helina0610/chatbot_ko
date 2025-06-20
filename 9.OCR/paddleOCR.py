from paddleocr import PaddleOCR


engine = PaddleOCR(lang="korean", ocr_version="PP-OCRv3")
result = engine.predict("data/9791168342569.jpg")
print(result)
for res in result:
    res.print()
    res.save_to_img(save_path="output")
    res.save_to_json(save_path="output")



