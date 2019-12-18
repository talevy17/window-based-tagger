from DataUtils import Parser, PreTrainedLoader, data_loader
from ModelTrainer import trainer_loop
import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_length, window_size, weights):
        super(Model, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
        self.input_dim = window_size * embedding_length
        self.non_linear = nn.Sequential(nn.Linear(self.input_dim, hidden_size), nn.Tanh())
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embed(x).view(-1, self.input_dim)
        x = self.non_linear(x)
        x = self.linear(x)
        return self.softmax(x)


def tagger_2():
    batch_size = 1000
    hidden_size = 100
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 10
    pretrained = PreTrainedLoader('./Data/pretrained/embeddings.txt', './Data/pretrained/words.txt')
    F2I = pretrained.get_dict()
    weights = pretrained.get_weights()
    weights = np.concatenate((weights, np.zeros((1, embedding_length))))
    vocab_train = Parser(window_size, F2I=F2I)
    vocab_train.parse_to_indexed_windows()
    L2I = vocab_train.get_l2i()
    I2L = vocab_train.get_i2l()
    vocab_valid = Parser(window_size, data_kind="dev", F2I=F2I, L2I=L2I)
    vocab_valid.parse_to_indexed_windows()
    output_size = len(L2I)
    model = Model(output_size, hidden_size, embedding_length, window_size, weights)
    model = trainer_loop(model, data_loader(vocab_train, batch_size),
                          data_loader(vocab_valid, batch_size), I2L, learning_rate, epochs)


if __name__ == "__main__":
    tagger_2()
