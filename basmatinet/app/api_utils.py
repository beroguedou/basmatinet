import base64
import io
import yaml
from PIL import Image
import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, make_response, request


class BasmatinetPrediction():

    def __init__(self, model_arch, model_path, config_path):
        # Load the model
        self.model_path = model_path
        self.config_path = config_path
        self.model_arch = model_arch

        # Inference image transformations
        self.transforms = A.Compose([
            A.Resize(width=224, height=224)
        ])

    @property
    def model(self):
        self.model_arch.load_state_dict(torch.load(
            self.model_path, map_location=torch.device('cpu')))
        self.model_arch.eval()
        return self.model_arch

    @property
    def labels_map_reverse(self):
        with open(self.config_path, 'r') as file:
            content = yaml.safe_load(file)
        return content['labels_map_reverse']

    def _load_image(self, image_b64):
        image_binary = base64.b64decode(image_b64)
        # Convert the base64 image to PIL Image object
        image_buf = io.BytesIO(image_binary)
        image = Image.open(image_buf)
        return image

    def _preprocess(self, image):
        X = np.asarray(image)
        X = self.transforms(image=X)['image']
        X = torch.from_numpy(X).permute(2, 0, 1).unsqueeze(0)
        X = X.float()
        return X

    def _predict(self, X):
        with torch.no_grad():
            out = self.model(X).squeeze(0)
            out = F.softmax(out, dim=-1)
            proba, index = torch.topk(out, 1)
            index = index.item()
            proba = round(proba.item(), 4)
        return proba, index

    def _post_process(self, proba, index):
        response = {'category': self.labels_map_reverse[index],
                    'probability': proba}
        return response

    def inference_pipeline(self, image_b64):
        # Load the image
        image = self._load_image(image_b64)
        # Preprocess it
        X = self._preprocess(image)
        # Go through the model and get a prediction
        proba, index = self._predict(X)
        # Post process the prediction and build a response
        response = self._post_process(proba, index)
        return response

    ####


def create_app(app, predictor):
    @app.route('/serving/predict', methods=['POST'])
    def prediction_pipeline():
        # Get the image in base 64 and decode it
        payload = request.form.to_dict(flat=False)
        image_b64 = payload['image'][0]
        # Pass it through the inference pipeline
        try:
            response = predictor.inference_pipeline(image_b64)
        except Exception as e:
            print(e)
        return make_response(jsonify(response))

    @app.route('/serving/healthcheck', methods=['GET'])
    def healthcheck():
        return 'Hello I am well !'
####
