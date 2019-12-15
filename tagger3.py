from Parser import Parser
from top_k import PreTrainedLoader
from tagger1 import iterate_model
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class Model(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, embedding_length, window_size, prefix_size, suffix_size):
        super(Model, self).__init__()
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab_size, embedding_length)
        self.concat_size = window_size * embedding_length
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.embed_prefix = nn.Embedding(prefix_size, embedding_length)
        nn.init.uniform_(self.embed_prefix.weight, -1.0, 1.0)
        self.embed_suffix = nn.Embedding(suffix_size, embedding_length)
        nn.init.uniform_(self.embed_suffix.weight, -1.0, 1.0)
        self.hidden = nn.Linear(self.concat_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        normal = self.embed(x[0]).view(-1, self.concat_size)
        prefix = self.embed_prefix(x[1]).view(-1, self.concat_size)
        suffix = self.embed_suffix(x[2]).view(-1, self.concat_size)
        data = normal + prefix + suffix
        data = self.hidden(data)
        data = torch.tanh(data)
        data = self.out(data)
        return self.softmax(data)


def make_loader(parser, batch_size):
    x = (parser.get_sentences(), parser.get_sentences_prefix(), parser.get_sentences_suffix())
    y = torch.from_numpy(np.array(parser.get_labels()))
    y = y.type(torch.long)
    # y = y.reshape(1, -1)[None, :, :]
    x = (torch.from_numpy(np.array(x[0])), torch.from_numpy(np.array(x[1])), torch.from_numpy(np.array(x[2])))
    x = [x[0].type(torch.long), x[1].type(torch.long), x[2].type(torch.long)]
    torch.cat((x[0][None, :, :], x[1][None, :, :], x[2][None, :, :]))
    return DataLoader(TensorDataset(x, y), batch_size, shuffle=True)


def tagger_3():
    batch_size = 100
    hidden_size = 100
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_train = Parser('./data/ner/train', window_size)
    vocab_train.parse_sentences(pos=False, with_prefix_suffix=True)
    L2I = vocab_train.get_l2i()
    F2I = vocab_train.get_f2i()
    I2L = vocab_train.get_i2l()
    vocab_valid = Parser('./data/ner/dev', window_size, F2I, L2I)
    vocab_valid.parse_sentences(pos=False, with_prefix_suffix=True)
    output_size = len(L2I)
    vocab_size = len(F2I)
    prefix_size = len(vocab_train.get_prefix_f2i())
    suffix_size = len(vocab_train.get_suffix_f2i())
    model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size, prefix_size, suffix_size)
    model = model
    model = iterate_model(model, make_loader(vocab_train, batch_size),
                          make_loader(vocab_valid, batch_size), learning_rate, epochs, I2L)


if __name__ == "__main__":
    tagger_3()
