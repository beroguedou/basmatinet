import numpy as np
import torch


def one_epoch_training(dataloader, model, criterion, optimizer, device):
    """ 
    """
    model.train()
    train_loss = 0.0
    for i, data in enumerate(dataloader):
        if i == 2:
            break
        # Get the inputs data and move to device
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # Zero (clear) the parameter gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(images)
        # Compute the loss
        loss = criterion(outputs, labels)
        #  Backward: compute the gradients
        loss.backward()
        #  Optimize: update the weigths
        optimizer.step()
        # Statistics
        train_loss += loss.item()
    return train_loss


def one_epoch_validation(dataloader, model, criterion, device):
    """ 
    """
    model.eval()
    
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i == 2:
                break
            # Get the inputs data and move to device
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # forward pass
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Statistics
            val_loss += loss.item()
    return val_loss
        


def all_epochs_training_and_validation(train_dataloader, val_dataloader, 
                                       model, criterion, optimizer, device,
                                       nb_epochs=20, early_stopping=5,
                                       model_name="basmatinet.pth"):
    """
    """
    counter = 0
    best_val_loss = np.inf
    for epoch in range(nb_epochs):
        train_loss = one_epoch_training(train_dataloader, model, criterion, optimizer, device)
        val_loss = one_epoch_training(val_dataloader, model, criterion, device)
        # Early stopping 
        if val_loss < best_val_loss:
            print("Epoch {} Loss decreased from {}  to ---->  {} \t Saving the model".format(epoch + 1, train_loss, val_loss))
            best_val_loss = val_loss
            # Save the model
            torch.save(model.state_dict(), model_name)
        else:
            counter += 1
            if counter == early_stopping:
                print("===="*3, " EARLY STOPPING ", "===="*3)
        # Print metrics
        print("Epoch, Train Loss {}  Val Loss {} ".format(epoch + 1, round(train_loss, 5) , round(val_loss)))