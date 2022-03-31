FROM python:3.6-slim
WORKDIR /usr/src/app
COPY requirements.txt api_utils.py models.py api.py app_config.yaml ./
RUN python -m pip install --upgrade pip && \ 
    pip install --no-cache-dir -r requirements.txt &&\
    pip3 install --no-cache-dir torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    torchaudio==0.9.0 -f \
    https://download.pytorch.org/whl/torch_stable.html

CMD ["python", "api.py"]