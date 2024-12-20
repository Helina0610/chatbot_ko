# 윈도우에서 리눅스(Oracle Linux)에서 Docker로 파이썬 프로젝트 배포하기

## 1. wsl 설치 및 oracle linux 설치
- window 관리자 cmd
	- `wsl --list --online`
	- `wsl --install OracleLinux9_1`
	- `wsl`
	- (선택사항) `wsl set defaul` , 찾아보기
	- update

## 2. docker 설치
- docker 설치 : docker 호환 프로그램으로 podman 설치
- `sudo dnf install podman` 

### 2.1. Docker 실행 전 사전준비
1. `requirements.txt`, `Dckerfile` 을 프로젝트 root 경로에 작성
2. `requirements.txt` 파일에 파이썬 환경에서 설치할 패키지 목록 작성
	- `torch` 는 리눅스에서 설치할 때 `--extra-index-url` 를 적어줘야한다
	```
	--extra-index-url https://download.pytorch.org/whl/cpu
	transformers
	torch
	Flask
	tokenizers
	protobuf
	sentencepiece
	pymupdf
	```

3. `Dckerfile` 도커에 설치될 파이썬 버전 명시 및 명령어 작성
	```
	// 파이썬 버전 설정
	FROM python:3.12
	// 프로젝트가 설치될 경로
	WORKDIR /usr/src/app
	// 패키지 리스트를 적은 requirements.txt 실행
	COPY requirements.txt ./
	RUN pip install --no-cache-dir -r requirements.txt
	COPY . .
	// 파이썬으로 실행할 파일 
	CMD [ "python", "./your-daemon-or-script.py" ]
	``` 
4. 프로젝트 폴더를 서버로 옮김

### 2.2. Docker 실행
1. `sudo dnf install docker`
2. 프로젝트 폴더로 이동
3. `docker build -t hongju-example .` 
	- 프로젝트를 `hongju-example` 로 명명
4. `docker run -it --rm -p 5000:5000 --name hongju hongju-example
	- `-p 5000:5000` : 도커의 5000 포트를 열겠다
	- `--name hongju` : 컨테이너의 이름을 `hongju` 로 명명 
	- `docker run -it --rm -p 5000:5000 --name chatbot_ko_container chatbot_ko_v3`

## 3. 방화벽 설정
- `sudo firewall-cmd --permanent --zone=public --add-port=5000/tcp`

