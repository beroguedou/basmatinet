from flask import Flask, jsonify, make_response
from models import RiceNet
import torch
from api_utils import (load_image, 
                       preprocess,
                       predict, 
                       post_process)

MODEL_PATH = "/data/basmatinet.pth"

# Map labels and categories
labels_dict_reverse = {0: 'Arborio',
                       1: 'Karacadag', 
                       2: 'Basmati', 
                       3: 'Jasmine',
                       4: 'Ipsala'}
# Load the model
model = RiceNet()
model.load_state_dict(torch.load(MODEL_PATH))
         
app = Flask(__name__)


@app.route('/image/predict/', methods=['POST'])
def prediction_pipeline():
    image = load_image()
    X = preprocess(image)
    proba, index = predict(X, model)
    response = post_process(proba, index, labels_dict_reverse)
    
    return make_response(jsonify(response))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)