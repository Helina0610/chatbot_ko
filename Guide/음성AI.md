# 음성AI 사용법

## Pyannote 모델을 로컬에서 사용하기
- Huggingface에서 Read 타입으로 token 발행
- `git clone` 시 로그인 팝업이 생성됨
- Huggingface Primary email, 발급받은 Token 으로 로그인
- [공식 튜토리얼](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb) 위의 방식으로 실행

## Whisper
- ffmpeg was not found but is required to load audio files from filename -> ffmpeg 설치
- [FFmpeg-Builds 윈도우즈 배포](https://www.gyan.dev/ffmpeg/builds/) 에서 `zip` 파일 다운로드
- 압축 해제 후, `ffmpeg/bin` 디렉토리를 시스템 PATH에 추가
- `cmd`에서 `ffmpeg -version` 명령으로 설치 확인
- 화자 분리 : https://github.com/pyannote/pyannote-audio
	- https://kjh1337.tistory.com/3