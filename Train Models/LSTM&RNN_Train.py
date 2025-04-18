import torch
import torch.nn as nn
import torch.optim as optim

# Define a unified model class that supports both LSTM and RNN
class SeqModel(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, output_size, num_layers):
        super(SeqModel, self).__init__()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if model_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif model_type == 'RNN':
            self.rnn_layer = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for output

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        if self.model_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initial cell state
            out, _ = self.rnn_layer(x, (h0, c0))  # Pass through LSTM layer
        else:
            out, _ = self.rnn_layer(x, h0)  # Pass through RNN layer

        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out

# Hyperparameters
input_size = 1  # For univariate time series, adjust for your data
hidden_size = 128
output_size = 1
num_layers = 2
num_epochs = 50
learning_rate = 0.001
model_type = 'LSTM'  # Choose between 'LSTM' or 'RNN'

# Dummy data (replace this with your actual sequence data)
sequence_length = 10
batch_size = 64
x_train = torch.randn(batch_size, sequence_length, input_size)
y_train = torch.randn(batch_size, output_size)

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SeqModel(model_type=model_type, input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    outputs = model(x_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
