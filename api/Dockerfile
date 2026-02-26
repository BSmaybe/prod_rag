FROM local-python:3.11-slim

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

ARG NEXUS_PYPI_URL="https://nexus.apps.devops.bcc.kz/repository/pypi-proxy/simple"
ARG NEXUS_HOST="nexus.apps.devops.bcc.kz"
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

COPY api/requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir --prefer-binary \
      --index-url "${NEXUS_PYPI_URL}" \
      --trusted-host "${NEXUS_HOST}" \
      -r requirements.txt

COPY api /app/api
COPY retriever /app/retriever
COPY utils /app/utils
COPY core /app/core
COPY etl /app/etl

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
