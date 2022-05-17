import pytest
import base64
import os
import torch
from basmatinet.ml.models import RiceNet
from basmatinet.app.api_utils import BasmatinetPrediction


MODEL_PATH = None
CONFIG_PATH = os.path.join(os.path.dirname(__file__), './app_config.yaml')
IMAGE_PATH = os.path.join(os.path.dirname(__file__), './arborio.jpg')


@pytest.fixture
def image():
    with open(IMAGE_PATH, 'rb') as img:
        img_b64 = base64.b64encode(img.read())
    return img_b64


@pytest.mark.usefixtures('image')
class TestBasmatinetPrediction():

    def setup_method(self):
        self.model_arch = RiceNet(pretrained=False)

        self.predictor = BasmatinetPrediction(self.model_arch,
                                              model_path=MODEL_PATH,
                                              config_path=CONFIG_PATH)

    def teardown_method(self):
        del self.predictor
        del self.model_arch

    def test_init_transforms(self):
        assert self.predictor.transforms

    def test_init_model(self, mocker):
        mocker.patch.object(BasmatinetPrediction, 'model',
                            return_value=self.model_arch)
        assert self.predictor.model is not None

    def test_load_image(self, image):
        assert self.predictor._load_image(image)

    @pytest.mark.dependency(depends=[test_load_image])
    def test_preprocess(self, image):
        output_shape = torch.Size([1, 3, 224, 224])
        image = self.predictor._load_image(image)
        X = self.predictor._preprocess(image)

        assert output_shape == X.shape

    def test_predict(self, mocker):
        X = torch.randn([1, 3, 224, 224])
        mocker.patch.object(BasmatinetPrediction, 'model',
                            return_value=self.model_arch(X))
        _, _ = self.predictor._predict(X)
        assert True

    def test_post_process(self):
        proba, index = 0.2373, 1
        response = self.predictor._post_process(proba, index)
        assert sorted(['category', 'probability']
                      ) == sorted(list(response.keys()))
