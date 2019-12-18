import torch
import torch.nn as nn
from ModelTrainer import trainer_loop
from DataUtils import Parser


class Model(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, embedding_length, window_size):
        super(Model, self).__init__()
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab_size, embedding_length)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.input_dim = window_size * embedding_length
        self.non_linear = nn.Sequential(nn.Linear(self.input_dim, hidden_size), nn.Tanh())
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embed(x).view(-1, self.input_dim)
        x = self.non_linear(x)
        x = self.linear(x)
        return self.softmax(x)


def tagger_1():
    batch_size = 1000
    hidden_size = 100
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 10
    train_data = Parser(window_size)
    train_data.parse_to_indexed_windows()
    label_to_idx = train_data.get_l2i()
    word_to_idx = train_data.get_f2i()
    idx_to_label = train_data.get_i2l()
    dev_data = Parser(window_size, 'pos', "train", word_to_idx, label_to_idx)
    dev_data.parse_to_indexed_windows()
    output_size = len(label_to_idx)
    vocab_size = len(word_to_idx)
    model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size)
    model = trainer_loop(model, train_data.data_loader(batch_size),
                          dev_data.data_loader(batch_size), idx_to_label, learning_rate, epochs)


if __name__ == "__main__":
    tagger_1()
