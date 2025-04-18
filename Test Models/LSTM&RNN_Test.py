import torch
import torch.nn as nn

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

# Hyperparameters (use the same as during training)
input_size = 1  # For univariate time series, adjust for your data
hidden_size = 128
output_size = 1
num_layers = 2
model_type = 'LSTM'  # Choose between 'LSTM' or 'RNN'

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SeqModel(model_type=model_type, input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)

# Load the saved model parameters (ensure you use the correct file path)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

# Dummy test data (replace this with your actual test data)
sequence_length = 10
batch_size = 64
x_test = torch.randn(batch_size, sequence_length, input_size).to(device)

# Testing the model
with torch.no_grad():
    test_output = model(x_test)
    print("Test Output:")
    print(test_output)
