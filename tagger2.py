from Parser import Parser
from top_k import PreTrainedLoader
from tagger1 import make_loader, iterate_model
import torch
import torch.nn as nn
import numpy as np


batch_size = 100
hidden_size = 100
embedding_length = 50
window_size = 5
learning_rate = 0.01
epochs = 10


class Model(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_length, window_size, weights):
        super(Model, self).__init__()
        torch.manual_seed(3)
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        self.concat_size = window_size * embedding_length
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.hidden = nn.Linear(self.concat_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        data = self.embed(x).view(-1, self.concat_size)
        data = self.hidden(data)
        data = torch.tanh(data)
        data = self.out(data)
        return self.softmax(data)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = PreTrainedLoader('./Data/pretrained/wordVectors.txt', './Data/pretrained/vocab.txt')
    F2I = pretrained.get_dict()
    F2I[''] = len(F2I)
    weights = pretrained.get_weights()
    weights = np.concatenate((weights, np.zeros((1, embedding_length))))
    vocab_train = Parser('./data/pos/train', window_size, F2I)
    vocab_train.parse_sentences(' ')
    L2I = vocab_train.get_l2i()
    I2L = vocab_train.get_i2l()
    vocab_valid = Parser('./data/pos/dev', window_size, F2I, L2I)
    vocab_valid.parse_sentences(' ')
    output_size = len(L2I)
    vocab_size = len(F2I)
    model = Model(output_size, hidden_size, embedding_length, window_size, weights)
    model = model
    model = iterate_model(model, make_loader(vocab_train, batch_size),
                          make_loader(vocab_valid, batch_size), learning_rate, epochs, I2L)
