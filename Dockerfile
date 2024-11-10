FROM python:3.11.3-slim

ENV PYTHONUNBUFFERED=0

WORKDIR /app
ADD . ./

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false --local

RUN poetry install
