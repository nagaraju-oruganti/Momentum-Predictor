import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNRegressor(nn.Module):
    def __init__(self, config, num_outputs):
        super(CNNRegressor, self).__init__()
        self.config = config
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(3 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_outputs)
        
        self.relu = nn.ReLU()

    def loss_fn(self, outputs, y):
        criterion = nn.MSELoss()
        loss = criterion(outputs, y)
        return loss
    
    def forward(self, inputs, y):
        # Shared layers
        x = self.pool1(self.relu(self.conv1(inputs)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        out = x.view(-1, 3 * 16 * 16)
        
        ## Train for target 1
        x = self.fc1(out)
        logits = self.fc2(x)
        
        loss = self.loss_fn(logits, y)
        
        return logits, loss
    
class HSLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(HSLSTMRegressor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(input_size, hidden_size)])
        self.lstm_layers.extend([nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.linear = nn.Linear(hidden_size, output_size)
        self.loss_fn = torch.nn.MSELoss()
        self.device = device
        
    def forward(self, x, y):
        h, c = [], []
        for _ in range(self.num_layers):
            h.append(torch.zeros(x.size(0), self.hidden_size).to(self.device))
            c.append(torch.zeros(x.size(0), self.hidden_size).to(self.device))
            
        for t in range(x.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer], c[layer] = self.lstm_layers[layer](x[:,t,:], (h[layer], c[layer]))
                else:
                    h[layer], c[layer] = self.lstm_layers[layer](h[layer-1], (h[layer], c[layer]))
                    
        out = self.linear(h[-1])
        
        if y is None:
            return out
        
        loss = self.loss_fn(out, y)
        return out, loss
        