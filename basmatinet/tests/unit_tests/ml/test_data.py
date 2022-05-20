import pytest
import torch
from basmatinet.ml import data


@pytest.fixture
def datapath(pytestconfig):
    return pytestconfig.getoption('datapath')


@pytest.mark.usefixtures('datapath')
class TestBasmatinetData():

    def test_dataset_image(self, datapath):
        # Expected output's shape
        expected = torch.Size([3, 224, 224])
        # Create a dataset
        dataset = data.RiceDataset(datapath,
                                   train=True,
                                   percentage=0.2)
        # Test one output shape
        output_image, _ = dataset[0]
        assert output_image.shape == expected

    def test_dataset_label(self, datapath):
        # Expected output's shape
        expected = 5
        # Create a dataset
        dataset = data.RiceDataset(datapath,
                                   train=True,
                                   percentage=0.2)
        # Test that output labels are 5
        _, output_label = dataset[0]
        assert (output_label.shape[0] == expected)

    @pytest.mark.dependency(depends=[test_dataset_image, test_dataset_label])
    def test_dataloader_images(self, datapath):
        # Expected output's shape
        expected = torch.Size([5, 3, 224, 224])
        # Dataset
        dataset = data.RiceDataset(datapath,
                                   train=True,
                                   percentage=0.2)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=5,
                                                 shuffle=True,
                                                 num_workers=2)
        # Test that one batch output image has dimension [batch_size, 3, 224, 224]
        for i, batch in enumerate(dataloader):
            images, _ = batch
            if i == 1:
                break
        assert images.shape == expected

    @pytest.mark.dependency(depends=[test_dataset_image, test_dataset_label])
    def test_dataloader_labels(self, datapath):
        # Expected output's shape
        expected = torch.Size([5, 5])
        # Dataset
        dataset = data.RiceDataset(datapath,
                                   train=True,
                                   percentage=0.2)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=5,
                                                 shuffle=True,
                                                 num_workers=2)
        # Test that one batch output labels has dimension [batch_size, 5]
        for i, batch in enumerate(dataloader):
            _, labels = batch
            if i == 1:
                break
        assert labels.shape == expected
