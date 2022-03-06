


def one_epoch_training(dataloader, model, criterion, optimizer, device):
    """ 
    """
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # Get the inputs data
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Statistics
        running_loss += loss.item()
        
        
        
def one_epoch_validation(dataloader, model, criterion, optimizer, device):
    """
    """
    with torch.no_grad():
        pass 


def all_epochs_training_and_validation(train_dataloader, model, criterion, optimizer, device, nb_epochs=20):
    """
    """
    for epoch in range(nb_epochs):
        one_epoch_trainer(dataloader, model, criterion, optimizer, device)