import base64
import io
import yaml
from PIL import Image
import albumentations as A
import numpy as np
import torch
from models import RiceNet
import torch.nn.functional as F

MODEL_PATH = "./basmatinet.pth"
CONFIG_PATH = "./app_config.yaml"

class BasmatinetPrediction():
    
    def __init__(self):
        # Load the model
        self.model = RiceNet()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        # Inference image transformations
        self.transforms = A.Compose([
                                A.Resize(width=224, height=224) 
                                ])
        
    @property
    def labels_map_reverse(self):
        with open(CONFIG_PATH, "r") as file:
            content = yaml.safe_load(file)
        return content["labels_map_reverse"]        

    def _load_image(self, image_b64):
        image_binary = base64.b64decode(image_b64)
        # Convert the base64 image to PIL Image object
        image_buf = io.BytesIO(image_binary)
        image = Image.open(image_buf)
        return image

    def _preprocess(self, image):
        X = np.asarray(image)
        X = self.transforms(image=X)["image"]
        X = torch.from_numpy(X).permute(2, 0, 1).unsqueeze(0)
        X = X.float()
        return X

    def _predict(self, X):
        out = self.model(X).squeeze(0)
        out = F.softmax(out, dim=-1)
        proba, index = torch.topk(out, 1)  
        index = index.item()
        proba = round(proba.item(), 4)
        return proba, index  
    

    def _post_process(self, proba, index):
        response = {"category": self.labels_map_reverse[index], 
                    "probability": proba}
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
    
    