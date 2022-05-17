FROM --platform=linux/amd64 python:3.6-slim
COPY basmatinet/app/requirements.txt /tmp
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

RUN groupadd -g 1000 mlgroup && useradd -u 1000 -g mlgroup mluser
RUN mkdir -p /usr/src/app && chown mluser:mlgroup /usr/src/app
USER mluser:mlgroup
WORKDIR /usr/src/app
COPY --chown=mluser:mlgroup basmatinet/app/api_utils.py basmatinet/app/api.py \
    basmatinet/app/basmatinet.pth basmatinet/app/app_config.yaml \
    basmatinet/app/models.py ./

EXPOSE 5001

CMD ["gunicorn", "--bind", ":5001", "api:app"]
