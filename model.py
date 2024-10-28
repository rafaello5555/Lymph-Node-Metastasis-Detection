# train.py





import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import models

# Hyperparameters
batch_size = 30
lr = 3e-4
n_epochs = 5
num_classes = 2

def initialize_model():
    # Load pretrained model
    model = models.resnet34(pretrained=True)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Define loss and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer
