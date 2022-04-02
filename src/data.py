import os
import functools
from PIL import Image
import numpy as np
import torch
import albumentations as A
import torch.nn.functional as F


class RiceDataset(torch.utils.data.Dataset):
    """ """
    def __init__(self, base_path, train=True, percentage=0.1):
        self.base_path = base_path
        self.train = train
        self.num_classes = len(os.listdir(self.base_path))
        

        # Take the last samples for validation and the first ones for training
        if self.train == True:
            self.nb_samples = [round(len(os.listdir(base_path+sub_path)) * percentage) for sub_path in os.listdir(base_path)]
            list_IDs = [os.listdir(base_path+sub_path)[0:self.nb_samples[i]] for i, sub_path in enumerate(os.listdir(base_path))]
            list_labels = [[sub_path for _ in range(self.nb_samples[i])] for i, sub_path in enumerate(os.listdir(base_path))]

        else:
            self.nb_samples = [round(len(os.listdir(base_path+sub_path)) * percentage) for sub_path in os.listdir(base_path)]
            list_IDs = [os.listdir(base_path+sub_path)[::-1][0:self.nb_samples[i]] for i, sub_path in enumerate(os.listdir(base_path))]
            list_labels = [[sub_path for _ in range(len(os.listdir(base_path+sub_path)) , self.nb_samples[i], -1)] for i, sub_path in enumerate(os.listdir(base_path))]

        self.list_IDs = functools.reduce(lambda x, y: x+y, list_IDs)
        self.list_labels = functools.reduce(lambda x, y: x+y, list_labels)
        
        # Validate if the file exists, if not delete it from the list
        self.labels_list_IDs = [(lab, id_) for (lab, id_) in zip(self.list_labels, self.list_IDs) if os.path.isfile(os.path.join(self.base_path, lab, id_))]
        self.list_labels  = [element[0] for element in self.labels_list_IDs]
        self.list_IDs = [element[1] for element in self.labels_list_IDs]
        
        # Augmentations
        if self.train == True:
            self.transforms = A.Compose([
                A.RandomCrop(width=224, height=224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2)
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(width=224, height=224) # Achanger plus tard
                ])
            
        self.labels_dict = dict(zip(os.listdir(self.base_path), range(self.num_classes)))
        self.labels_dict_reverse = dict(zip(range(self.num_classes), os.listdir(self.base_path)))
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        # Load data and label
        label = self.list_labels[index]
        image_path = self.base_path+label+'/'+ID
        X = Image.open(image_path)
        X = np.asarray(X)
        # Augmentations while training
        X = self.transforms(image=X)["image"]
           
        X = torch.from_numpy(X).permute(2, 0, 1)
        X = X.float()
        y = self.labels_dict[label]
        y = F.one_hot(torch.tensor(y), num_classes=self.num_classes)
        y = y.float()
        
        return X, y