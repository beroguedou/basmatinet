from flask import Flask, jsonify, make_response, request
from models import RiceNet
from api_utils import BasmatinetPrediction


MODEL_PATH = './basmatinet.pth'
CONFIG_PATH = './app_config.yaml'

model_arch = RiceNet(pretrained=False)
predictor = BasmatinetPrediction(model_arch=model_arch,
                                 model_path=MODEL_PATH,
                                 config_path=CONFIG_PATH)

app = Flask(__name__)


@app.route('/serving/predict', methods=['POST'])
def prediction_pipeline():
    # Get the image in base 64 and decode it
    payload = request.form.to_dict(flat=False)
    image_b64 = payload['image'][0]
    # Pass it through the inference pipeline
    response = predictor.inference_pipeline(image_b64)
    return make_response(jsonify(response))


@app.route('/serving/healthcheck', methods=['GET'])
def healthcheck():
    return 200


if __name__ == '__main__':
    app.run()
