import torch.nn as nn
import torch.nn.functional as F

# Define the CNN architecture

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv Layer 1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # Conv Layer 2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Conv Layer 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)

        # Linear Layers
        self.fc1 = nn.Linear(4 * 4 * 64, 500)
        self.fc2 = nn.Linear(500, 10)

        # dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Perform convolutions on input with max pool layer and relu activation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten view of the convolutional layer
        x = x.view(-1, 4 * 4 * 64)

        # Add dropout layer
        x = self.dropout(x)

        # Add first linear layer
        x = F.relu(self.fc1(x))

        # Add another dropout layer
        x = self.dropout(x)

        # Add second linear layer
        x = self.fc2(x)

        return x