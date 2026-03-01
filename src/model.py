import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_filters, kernel_size, num_units):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        conv_output_size = num_filters * 14 * 14
        
        self.fc1 = nn.Linear(conv_output_size, num_units)
        self.fc2 = nn.Linear(num_units, 10)

 
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x