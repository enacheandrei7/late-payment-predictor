# syntax=docker/dockerfile:1
FROM python:3.9-slim-buster
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r requirements.txt
COPY . .
EXPOSE $PORT
ENV FLASK_APP=./api/app.py
CMD ["flask", "run", "--host", "0.0.0.0"]