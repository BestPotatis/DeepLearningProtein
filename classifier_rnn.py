import torch.nn as nn

# create RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        # rnn
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, 
                            batch_first = True)

        # define fully connected layers
        self.linear = nn.Linear(hidden_size, num_classes, bias = False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # return output and last hidden state
        x, (h, c) = self.lstm(x)
        
        # fully-connected output layer
        x = self.softmax(self.linear(x))
        
        return x
