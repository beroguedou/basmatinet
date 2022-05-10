import os
import json
import numpy as np
import torch


def artefacts_loader(model_path, labels_json_path):
    # Load the model
    model = torch.load(model_path)
    model.eval()
    # Load the labels
    labels = 1  # just a placeholder
    return model, labels


def preprocess(image_path):
    image = 1  # just a placeholder
    return image


def predict(image, model):
    """
    """
    with torch.no_grad():
        output = model(image)
    return output


def postprocess():
    pass


def inference():
    pass
