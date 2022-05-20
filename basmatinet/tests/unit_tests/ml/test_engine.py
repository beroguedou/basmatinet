import pytest
import torch
from basmatinet.ml import (
    data,
    engine,
    models
)


@pytest.fixture
def datapath(pytestconfig):
    return pytestconfig.getoption('datapath')


@pytest.mark.usefixtures('datapath')
class TestBasmatinetEngine():

    def setup_method(self):

        # Other stufs
        self.model = models.RiceNet()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.device = torch.device('cpu')

    def teardown_method(self):
        del self.model
        del self.criterion
        del self.optimizer
        del self.device

    def test_one_epoch_training(self, datapath):
        # Dataset

        dataset = data.RiceDataset(datapath,
                                   train=True,
                                   percentage=0.2)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=5,
                                                 shuffle=True,
                                                 num_workers=2)
        engine.one_epoch_training(dataloader,
                                  self.model,
                                  self.criterion,
                                  self.optimizer,
                                  self.device,
                                  breakpoint=2)
        assert True

    def test_one_epoch_validation(self, datapath):

        val_dataset = data.RiceDataset(datapath,
                                       train=True,
                                       percentage=0.2)

        # Dataloader
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=5,
                                                     shuffle=True,
                                                     num_workers=2)

        engine.one_epoch_validation(val_dataloader,
                                    self.model,
                                    self.criterion,
                                    self.device,
                                    breakpoint=2)
        assert True
