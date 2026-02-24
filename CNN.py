import torch.nn as nn
import torch
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data = "./data"

class CNN(nn.Module):
    def __init__(self, num_classes=10,in_channels=1, features_fore_linear=25088, dataset=None):
        super().__init__()

        conv_stride = 1
        pool_stride = 2
        conv_kernel = 3
        pool_kernel = 2
        dropout_rate = 0.25
        momentum = 0.9
        weight_decay = 0.0001
        learning_rate = 0.001

        conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=conv_kernel, padding=1),
        nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=conv_kernel, padding=1),
        nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
        nn.ReLU(),
        nn.Flatten(),
        
            ).to(device)

        self.features = conv1

        self.classifier = nn.Sequential(
            nn.Linear(in_features=features_fore_linear, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
            nn.ReLU(),
            nn.Flatten(),
        ).to(device)
    
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(list(self.features.parameters()) + list(self.classifier.parameters()), lr=learning_rate)
    
        self.dataset = dataset

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def train_model(self, train_dataloader, num_epochs=10, val_dataloader=None):
        # Call .train() on self to turn on dropout
        self.train()

        # To hold accuracy during training and testing
        train_accs = []
        test_accs = []

        for epoch in range(num_epochs):
            
            epoch_acc = 0

            for inputs, targets in tqdm(train_dataloader):
                logits = self(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Keep track of training accuracy
                epoch_acc += (torch.argmax(logits, dim=1) == targets).sum().item()
            train_accs.append(epoch_acc / len(train_dataloader.dataset))

            # If val_dataloader, evaluate after each epoch
            if val_dataloader is not None:
                # Turn off dropout for testing
                self.eval()
                acc = self.eval_model(val_dataloader)
                test_accs.append(acc)
                print(f"Epoch {epoch} validation accuracy: {acc}")
                # turn on dropout after being done
                self.train()
        
        return train_accs, test_accs
    
    def eval_model(self, test_dataloader):
        self.eval()
        total_acc = 0
        for inputs, labels in test_dataloader:

            logits = self(inputs)
            total_acc += (torch.argmax(logits, dim=1) == labels).sum().item()

        total_acc = total_acc / len(test_dataloader.dataset)

        return total_acc
    
    