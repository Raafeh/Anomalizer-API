FROM python:3.9.12

WORKDIR /app

COPY . /app

RUN pip install tensorflow

RUN pip install -r requirements.txt

CMD uvicorn main:app --port=8000 --host=0.0.0.0