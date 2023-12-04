import torch.nn as nn

# create RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        
        self.dropout = nn.Dropout(p = 0.2)
        self.hidden_size = hidden_size
        # rnn
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                            batch_first = True)

        # define fully connected layers
        self.linear = nn.Linear(hidden_size, num_classes, bias = False)

    def forward(self, x):
        # return output and last hidden state
        x, (h, c) = self.lstm(self.dropout(x))
        
        # fully-connected output layer
        x = self.linear(x)
        
        return x
