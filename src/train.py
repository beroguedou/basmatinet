import argparse
import data 
import models 
import engine
import torch


training_parser = argparse.ArgumentParser(description="Parameters to train a basmatinet model.")
training_parser.add_argument("datapath", type=str, help="Path to training dataset.")
training_parser.add_argument("--batch-size", type=int, default=16, 
                             help="Batch size for training and validation.")
training_parser.add_argument("--workers", type=int, default=8, 
                             help="Number of cpu cores for multiprocessing")
training_parser.add_argument("--early-stopping", type=int, default=5, 
                             help="Number of epochs to wait before stopping the training if no improvement noticed.")
training_parser.add_argument("--nb-epochs", type=int, default=200, help="Numbers of epochs to train.")
training_parser.add_argument("--percentage", type=float, default=0.1, help="Validation dataset size.")
training_parser.add_argument("--cuda", action="store_true", help="If True GPU")

args = training_parser.parse_args()

# Defining the parameters
datapath = args.datapath
batch_size = args.batch_size
workers = args.workers
early_stopping = args.early_stopping
nb_epochs = args.nb_epochs
percentage = args.percentage

# Setting the device cuda or cpu
if (torch.cuda.is_available() == True) and (args.cuda == True):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

    
# Datasets
train_dataset = data.RiceDataset(datapath, 
                                 train=True, 
                                 percentage=(1 - percentage)
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
engine.all_epochs_training_and_validation(train_dataloader, 
                                          val_dataloader, 
                                          net, 
                                          criterion, 
                                          optimizer, 
                                          device,
                                          nb_epochs=nb_epochs, 
                                          early_stopping=early_stopping,
                                          model_name="basmatinet.pth")
