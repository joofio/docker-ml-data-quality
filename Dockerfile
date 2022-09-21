#FROM python:3.9-slim-buster
#FROM ubuntu:18.04
FROM python:3.9-slim

RUN mkdir /app
RUN  apt-get update
RUN apt-get install zip -y
RUN apt-get -y install gcc

COPY requirements.txt /app

COPY *.py /app/
COPY *.json /app/
COPY *.sav /app/
COPY *.xml /app/
COPY *.pickle /app/
COPY *.zip /app/


WORKDIR /app
RUN unzip gritbot.zip
RUN pip install -r requirements.txt
RUN pip uninstall numpy --yes
RUN pip install numpy==1.20 

EXPOSE 5000
CMD uvicorn app:app --host=0.0.0.0 --port=5000
