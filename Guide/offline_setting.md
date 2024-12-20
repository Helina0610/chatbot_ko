# 1. 폐쇄망 환경
## **1.1. 인터넷 환경에서 사전준비**
> [!Note]
> - 파이썬 도커 이미지, 파이썬 패키지를 인터넷이 되는 환경에서 미리 다운로드 한다.
> - redhat 리눅스에서는 docker를 설치하는 대신 기본적으로 설치된 podman 을 사용할 수 있다.
> - 전체 과정
> 	1. 인터넷 환경에서 필요한 파일(파이썬 도커 이미지, 패키지) 다운로드
> 	2. 파일들 서버로 이동 후 불러오기
> 	3. 도커 이미지 생성 및 컨테이너 실행
### **1.1.1. 관련 패키지 다운로드**
1. `Dockerfile`, `requirements.txt`  프로젝트 root 에 작성
2. `requirements.txt` 파일에 패키지의 버전 명시
	```
	--extra-index-url https://download.pytorch.org/whl/cpu
	transformers==4.47.0
	torch==2.5.1
	Flask==3.1.0
	tokenizers==0.21.0
	protobuf==5.29.1
	sentencepiece==0.2.0
	pymupdf==1.25.1
	```
2. `pip download -r requirements.txt -d ./packages` 
	- `requirements.txt` 에 적힌 라이브러리를 `./packages` 폴더에 다운로드
3. `Dockerfile`
	```Dockerfile
	FROM python:3.12
	
	WORKDIR /usr/src/app
	
	COPY requirements.txt ./
	RUN pip install --no-cache-dir --find-links=/packages -r requirements.txt
	
	COPY . .
	
	CMD [ "python", "./index.py" ]
	```

### **1.1.2. 파이썬 도커 이미지 다운로드**
- 도커 파이썬 이미지 다운로드 :  [파이썬 도커 이미지](https://hub.docker.com/_/python/tags?name=3.12)
	1. `docker pull python:3.12` : 파이썬 도커 이미지 다운로드
	2. `docker save python:3.12 | gzip > docker-python.tar.gz` : 파이썬 도커 이미지 저장 및 압축
	
## **1.2. 폐쇄망에서 파이썬 이미지 불러오기**
1. 사전 준비한 파일들을 서버로 이동
2. redhat linux 에서 `Docker` 대신 `podman` 사용할 수 있다.
3. `podman import docker-python.tar.gz python:3.12`
4. `podman image ls` 파이썬 이미지 확인
## **1.3. 도커 이미지 생성 및 컨테이너 실행**
1. `cd 프로젝트 경로` , 프로젝트 폴더로 이동
2. `docker build -t chatbot_ko_image .` 
	- 생성할 도커 이미지를 `chatbot_ko_image` 로 명명
3. `docker run -it --rm -p 5000:5000 --name chatbot_ko_container chatbot_ko_image`
	- `docker run` : Docker 컨테이너를 실행하는 명령어
	- `--rm` : 컨테이너가 종료되면 자동으로 삭제
	- `-p 5000:5000` : 도커의 5000 포트를 열겠다
	- `--name chatbot_ko_container` : 컨테이너의 이름을 `chatbot_ko_container` 로 명명
	- `chatbot_ko_image` : 실행할 Docker 이미지의 이름
## 1.4. 방화벽 설정
- Flask 5000 포트 오픈
- `$ sudo firewall-cmd --permanent --zone=public --add-port=5000/tcp`
