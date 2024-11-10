FROM python:3.13-alpine

ENV PYTHONUNBUFFERED=0

RUN apk update && apk add --no-cache netcat-openbsd

WORKDIR /app
ADD . ./

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false --local

RUN poetry install

RUN apk add --no-cache --virtual .build-deps gcc musl-dev libffi-dev \
    && apk del .build-deps


RUN chmod +x ./run.sh

CMD ["./run.sh"]