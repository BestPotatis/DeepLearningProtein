import torch.nn as nn
import torch

# create RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layer):
        super(RNN, self).__init__()
        
        self.dropout = nn.Dropout(p = 0.2)
        self.hidden_size = hidden_size
        self.num_layers = num_layer
        # rnn
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                            batch_first = True, bidirectional = True, num_layers = num_layer)

        # define fully connected layers
        self.linear = nn.Linear(hidden_size*2, num_classes, bias = False)

    def forward(self, x):
        # return output and last hidden state
        x, (h, c) = self.lstm(self.dropout(x))
        
        # fully-connected output layer
        x = self.linear(x)
        
        return x

if __name__ == "__main__":
    m = RNN(1, 5, 3)
    print(m(torch.Tensor([[1]])))