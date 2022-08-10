#FROM python:3.9-slim-buster
#FROM ubuntu:18.04
FROM python:3.9-slim

RUN mkdir /app

COPY requirements.txt /app

COPY *.py /app/
COPY *.json /app/
COPY *.sav /app/
COPY *.xml /app/


WORKDIR /app
RUN pip install -r requirements.txt
RUN pip uninstall numpy --yes
RUN pip install numpy==1.20 

EXPOSE 5000
CMD uvicorn app:app --host=0.0.0.0 --port=5000
