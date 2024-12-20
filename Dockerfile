FROM python:3.12

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir --find-links=/packages -r requirements.txt

COPY . .

CMD [ "python", "./index.py" ]
