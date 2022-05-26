import pytest
import base64
import os
from basmatinet.app.models import RiceNet
from basmatinet.app.api_utils import BasmatinetPrediction, create_app
from flask import Flask

MODEL_PATH = None
CONFIG_PATH = os.path.join(os.path.dirname(__file__), './app_config.yaml')
IMAGE_PATH = os.path.join(os.path.dirname(__file__), './arborio.jpg')


@pytest.fixture
def image():
    with open(IMAGE_PATH, 'rb') as img:
        img_b64 = base64.b64encode(img.read())
    return img_b64


@pytest.mark.usefixtures('image')
class TestBasmatinetApi():

    def setup_method(self):
        self.hc_url = 'http://0.0.0.0:5001/serving/healthcheck'
        self.api_url = 'http://0.0.0.0:5001/serving/predict'
        self.model_arch = RiceNet(pretrained=False)
        self.predictor = BasmatinetPrediction(model_arch=self.model_arch,
                                              model_path=MODEL_PATH,
                                              config_path=CONFIG_PATH)
        # Create manually a flask test client
        self.app = Flask(__name__)
        create_app(self.app, self.predictor)
        self.client = self.app.test_client()

    def teardown_method(self):
        del self.predictor
        del self.model_arch
        del self.hc_url
        del self.api_url

    def test_healthcheck(self):
        assert self.client.get(self.hc_url).status_code == 200

    def test_prediction_model(self, mocker, image):
        probas, index = 0.2373, 1
        expected = sorted(['category', 'probability'])
        mocker.patch.object(BasmatinetPrediction, 'model',
                            return_value=self.model_arch)
        mocker.patch.object(BasmatinetPrediction, '_predict',
                            return_value=(probas, index))
        payload = {'image': image}
        response = self.client.post(
            self.api_url, data=payload).get_data().decode('utf-8')
        assert self.client.post(self.api_url, data=payload).status_code == 200
        assert expected == sorted(list(eval(response).keys()))
