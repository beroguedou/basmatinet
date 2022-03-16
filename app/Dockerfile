FROM python:3.6-slim
WORKDIR /usr/src/app
COPY . .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    torchaudio==0.9.0 -f \
    https://download.pytorch.org/whl/torch_stable.html

VOLUME /data

EXPOSE 5000
CMD ["python", "api.py"]