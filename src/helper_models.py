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
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, dropout_prob = 0.0, fine_tune = False):
        super(HSLSTMRegressor, self).__init__()
        self.fine_tune = fine_tune
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(input_size, hidden_size)])
        self.lstm_layers.extend([nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout_prob) for _ in range(num_layers)])
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
                    
                if self.fine_tune and layer != self.num_layers - 1:
                    h[layer].detach_()
                    c[layer].detach_()
                    
                if layer == self.num_layers - 1:
                    h[layer] = self.dropout_layers[layer](h[layer])
                    
        out = self.linear(h[-1])
        
        if y is None:
            return out
        
        loss = self.loss_fn(out, y)
        return out, loss
        
        
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, num_heads, num_layers, hidden_dim):
        super(TransformerModel, self).__init__()

        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = self.get_positional_encoding(seq_length, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.loss_fn = torch.nn.MSELoss()
        
    def forward(self, x, y = None):
        # Embed input and add positional encodings
        x = self.embedding(x) + self.positional_encoding
        x = x.permute(1, 0, 2)  # Reshape for transformer
        
        # Pass through the transformer encoder
        encoder_output = self.encoder(x)
        
        # Project to output dimension
        out = self.decoder(encoder_output)
        if y is None:
            return out
        
        loss = self.loss_fn(out, y)
        return out, loss
    
    def get_positional_encoding(self, seq_length, hidden_dim):
        # Generate sinusoidal positional encodings
        position = torch.arange(0, seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        pos_enc = torch.zeros(seq_length, hidden_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc
