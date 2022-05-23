import os
from flask import Flask
from models import RiceNet
from api_utils import BasmatinetPrediction, create_app


MODEL_PATH = os.environ['MODEL']  # './basmatinet.pth'
CONFIG_PATH = './app_config.yaml'

model_arch = RiceNet(pretrained=False)
predictor = BasmatinetPrediction(model_arch=model_arch,
                                 model_path=MODEL_PATH,
                                 config_path=CONFIG_PATH)

app = Flask(__name__)

create_app(app, predictor)


if __name__ == '__main__':
    app.run()
