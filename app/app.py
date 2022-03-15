import sys
import os
import pathlib

parent = pathlib.Path(__file__).resolve().parent.parent
parent = os.path.join(parent, "src")
sys.path.append(parent)

import base64
import io
from PIL import Image
from flask import Flask, jsonify, make_response, request
from models import RiceNet
import albumentations as A
import numpy as np
import torch

MODEL_PATH = "/home/beranger/basmatinet/app/basmatinet.pth"
# Map labels and categories
labels_dict_reverse = {0: 'Arborio', 1: 'Karacadag', 2: 'Basmati', 3: 'Jasmine', 4: 'Ipsala'}
# Load the model
model = RiceNet()
model.load_state_dict(torch.load(MODEL_PATH))

transforms = A.Compose([
                A.Resize(width=224, height=224) # Achanger plus tard
                ])

app = Flask(__name__)

@app.route('/image/predict/', methods=['POST'])
def predict():
    # Get the image in base 64 and decode it
    payload = request.form.to_dict(flat=False)
    image_b64 = payload["image"][0]
    image_binary = base64.b64decode(image_b64)
    # Convert the base64 image to PIL Image object
    image_buf = io.BytesIO(image_binary)
    image = Image.open(image_buf)
    # Preprocess the image and predict with the model
    X = np.asarray(image)
    X = transforms(image=X)["image"]
    X = torch.from_numpy(X).permute(2, 0, 1).unsqueeze(0)
    X = X.float()
    out = model(X).squeeze(0)
    proba, index = torch.topk(out, 1)    
    # through the inference function
    # Format and return the response
    response = {"category": labels_dict_reverse[index.item()], 
                "probability": round(proba.item(), 4)}
    
    return make_response(jsonify(response))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)