# Flask 선택 사유
기존 `java web` 에 붙여서 써야하기 때문에 가볍고 확장성이 넓은 파이썬 프레임워크가 필요했습니다. `Flask` 는 풀 스택 웹 프레임워크인 `django` 보다 가볍고 다양한 웹 엔진과 붙여서 쓸 수 있어서 선택하게 되었습니다,
# Flask 설치 방법
- 파이썬이 설치되었다고 가정하고 시작하겠습니다
- 파이썬 프로젝트 설치 경로 : `"E:\\\\hj_code\\\\Python\\\\flask"` 라 가정하겠습니다.
- 파이썬 프로젝트는 vscode 를 사용하여 실행하겠습니다.
## 1. 파이썬 가상환경 구성하기

- 파이썬 프로젝트 경로를 vscode 로 실행합니다.
- 터미널을 열고, 다음 코드를 작성합니다
    ```
    python -m venv .venv
    ```
- 위의 코드를 실행하면 프로젝트에 `.venv` 폴더가 생성됩니다.
- `ctrl`+ `Shift`+`p` 를 눌러 `Select Interpreter` 를 선택합니다
- 생성된 가상환경 폴더명 `.venv` 를 선택합니다
- 다시 터미널을 실행 시키면 아래와 같이 가상환경으로 실행이 됩니다.

# 2. Flask 설치
```powershell
pip install Flask
```

# 3. 예시
```python
from flask import Flask, jsonify, request
  
@app.route("/", methods=['GET'])
def hello():
  return "hello"
  
if __name__ == "__main__":
    app.run(debug=True)
```

## 3.1 Postman 활용
- vscode 오른쪽 ▶️ (Run Python File) 버튼으로 실행합니다
- postman 에서 실행합니다