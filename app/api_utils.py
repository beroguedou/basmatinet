import base64
import io
from PIL import Image
from flask import request
import albumentations as A
import numpy as np
import torch


transforms = A.Compose([
                A.Resize(width=224, height=224) # Achanger plus tard
                ])

def load_image():
    # Get the image in base 64 and decode it
    payload = request.form.to_dict(flat=False)
    image_b64 = payload["image"][0]
    image_binary = base64.b64decode(image_b64)
    # Convert the base64 image to PIL Image object
    image_buf = io.BytesIO(image_binary)
    image = Image.open(image_buf)
    return image

def preprocess(image):
    X = np.asarray(image)
    X = transforms(image=X)["image"]
    X = torch.from_numpy(X).permute(2, 0, 1).unsqueeze(0)
    X = X.float()
    return X

def predict(X, model):
    out = model(X).squeeze(0)
    proba, index = torch.topk(out, 1)  
    index = index.item()
    proba = round(proba.item(), 4)
    return proba, index  
    

def post_process(proba, index, labels_dict_reverse):
    response = {"category": labels_dict_reverse[index], 
                "probability": proba}
    return response

