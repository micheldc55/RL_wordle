import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        return self.fc3(x)
