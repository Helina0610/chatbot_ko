# Local AI 모델 사용하기

- python : 3.12.8
- Flask : 3.1.0
- torch : 2.3.1+cu118
- transformers : 4.46.3

# AI Model 및 라이브러리리

## AI 모델
- [Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M)
- [deepset/xlm-roberta-large-squad2](https://huggingface.co/deepset/xlm-roberta-large-squad2)

## 텍스트 전처리 라이브러리 (한국어)
- 형태소 분석 및 불용어 제거
	- [PyKomoran](https://pydocs.komoran.kr/firststep/installation.html)
	- [KoNLPy](https://konlpy.org/ko/latest/index.html)
- 띄어쓰기 교정
	- [pykospacing](https://github.com/haven-jeon/PyKoSpacing)
- 한국어 문장 분리 도구
	- [kiwipiepy](https://github.com/bab2min/kiwipiepy)

## 임베딩 모델
- [snunlp/KR-SBERT-V40K-klueNLI-augSTS](https://huggingface.co/snunlp/KR-SBERT-V40K-klueNLI-augSTS)

# 목차
0. 텍스트 전처리
1. 로컬 gguf 모델 실행
2. 로컬 QA모델 실행
3. LoRA튜닝_llamafactory
4. gguf 파일 변환
5. gguf 모델 활용
6. 모델 테스트
7. RAG
8. 음성AI
9. OCR

# Guide
미래의 나를 위해 작성된 가이드
- [Flask 설치](Guide/Flask_설치.md)
- [Local Al Model 사용하기](Guide/Local_AI_%20Model_사용하기.md)
- [서버에 도커 설치 및 파이썬 프로젝트 실행](Guide/Docker_Python.md)
- [폐쇄망에서 AI 프로젝트 실행하기](Guide/offline_setting.md)
- [음성AI 사용하기](Guide/음성AI.md)