# syntax=docker/dockerfile:1
FROM local-python:3.11-slim

ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# 1) Ставим Python-зависимости
COPY api/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

# 2) Копируем код по папкам
COPY api /app/api
COPY retriever /app/retriever
COPY utils /app/utils
COPY core /app/core
COPY etl /app/etl

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
