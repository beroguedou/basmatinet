import pytest
import os
from basmatinet.ml import (
    data,
    models,
    engine
)
import torch
import logging
from rich.logging import RichHandler
from basmatinet.ml.logging_config import logging_config
from basmatinet.ml.utils import ParameterError


logging.config.dictConfig(logging_config)
logger = logging.getLogger('root')
logger.handlers[0] = RichHandler(markup=True)


# Defining the parameters
batch_size = 3
workers = 2
early_stopping = 2
nb_epochs = 10
percentage = 0.2
cuda = True


@pytest.fixture()
def datapath(pytestconfig):
    return pytestconfig.getoption('datapath')


def test_integration_train(datapath):

    # Checking if parameters are valid
    if (nb_epochs is None) or (nb_epochs <= 0):
        message = 'Number of epochs should be at least 01 or greater .'
        logger.critical(message)
        raise ParameterError(nb_epochs, message)

    if early_stopping >= nb_epochs:
        message = 'Early_stopping be lesser tahn number of epochs .'
        logger.critical(message)
        raise ParameterError(early_stopping, message)

    if percentage < 1.0:
        if percentage > 0.5:
            message = 'Percentage of validation data should be between 0.0 and 0.5 .'
            logger.error(message)
        if percentage < 0:
            message = 'Percentage of validation data should be positive .'
            logger.critical(message)
            raise ParameterError(percentage, message)
    else:
        message = 'Percentage should be lesser than 1.0 .'
        logger.critical(message)
        raise ParameterError(percentage, message)

    # Setting the device cuda or cpu
    if (torch.cuda.is_available() == True) and (cuda == True):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Datasets
    train_dataset = data.RiceDataset(datapath,
                                     train=True,
                                     percentage=percentage
                                     )
    val_dataset = data.RiceDataset(datapath,
                                   train=False,
                                   percentage=percentage
                                   )

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=workers)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=workers)

    # Intanciating a model architecture
    net = models.RiceNet()
    # Declaring Criterion and Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    net.to(device)

    # Training
    engine.all_epochs_training_and_validation(logger,
                                              train_dataloader,
                                              val_dataloader,
                                              net,
                                              criterion,
                                              optimizer,
                                              device,
                                              nb_epochs=nb_epochs,
                                              early_stopping=early_stopping,
                                              model_name='fake_basmatinet.pth',
                                              breakpoint=3)
    # To see if the training produced the expected output
    assert os.path.isfile('basmatinet/app/fake_basmatinet.pth')


def test_presence_model_to_deploy():
    # To see if there is at least one model to deploy, change "basmatinet.pth"
    # to model's name if needed.
    assert os.path.isfile('basmatinet/app/basmatinet.pth')
