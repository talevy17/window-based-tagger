from torch.autograd import Variable
from DataUtils import DataReader, FromPreTrained
from ModelTrainer import trainer_loop, predict
import torch
import torch.nn as nn
import numpy as np
import sys


prefixes = []
suffixes = []
PRE2I = {}
SUF2I = {}


class Model(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, embedding_length, window_size, prefix_size, suffix_size,
                 weights=np.asarray([]), embedding_freeze=False):
        super(Model, self).__init__()
        torch.manual_seed(3)
        if weights.any():
            self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=embedding_freeze)
        else:
            self.embed = nn.Embedding(vocab_size, embedding_length)
            nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.embed_prefix = nn.Embedding(prefix_size, embedding_length)
        nn.init.uniform_(self.embed_prefix.weight, -1.0, 1.0)
        self.embed_suffix = nn.Embedding(suffix_size, embedding_length)
        nn.init.uniform_(self.embed_suffix.weight, -1.0, 1.0)
        self.input_dim = window_size * embedding_length
        self.non_linear = nn.Sequential(nn.Linear(self.input_dim, hidden_size), nn.Tanh())
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        prefix_vec, suffix_vec = x.numpy().copy(), x.numpy().copy()
        prefix_vec = prefix_vec.reshape(-1)
        suffix_vec = suffix_vec.reshape(-1)

        for i, pre in enumerate(prefix_vec):  # replace each word index with it's prefix index
            prefix_vec[i] = get_prefix_index_by_word_index(pre)
        for i, suf in enumerate(suffix_vec):  # replace each word index with it's suffix index
            suffix_vec[i] = get_suffix_index_by_word_index(suf)

        # return to the shape of x and make variables
        suffix_vec = torch.from_numpy(suffix_vec.reshape(x.data.shape))
        prefix_vec = torch.from_numpy(prefix_vec.reshape(x.data.shape))
        prefix_vec, suffix_vec = prefix_vec.type(torch.LongTensor), suffix_vec.type(torch.LongTensor)
        prefix_vec, suffix_vec = Variable(prefix_vec), Variable(suffix_vec)

        normal_embed = self.embed(x)
        prefix_embed = self.embed_prefix(prefix_vec)
        suffix_embed = self.embed_suffix(suffix_vec)
        # sum embedding vectors of word

        x = (prefix_embed + normal_embed + suffix_embed).view(-1, self.input_dim)
        x = self.non_linear(x)
        x = self.linear(x)
        return self.softmax(x)


def get_prefix_index_by_word_index(index):
    return PRE2I[prefixes[index]]


def get_suffix_index_by_word_index(index):
    return SUF2I[suffixes[index]]


def create_dictionaries(prefix_size, suffix_size, window_sentences, i2f):
    global PRE2I, SUF2I, prefixes, suffixes
    words = [word for sublist in window_sentences for word in sublist]
    prefixes = [i2f[word][: prefix_size] for word in words]
    suffixes = [i2f[word][-suffix_size:] for word in words]
    PRE2I = {pre: i for i, pre in enumerate(sorted(set(prefixes)))}
    SUF2I = {suf: i for i, suf in enumerate(sorted(set(suffixes)))}


def normal_loader(data_type, window_size):
    train_data = DataReader(window_size, data_type=data_type, to_lower=True)
    L2I = train_data.get_l2i()
    F2I = train_data.get_f2i()
    I2L = train_data.get_i2l()
    I2F = train_data.get_i2f()
    return train_data, F2I, L2I, I2L, I2F, np.asarray([])


def pre_trained_loader(data_type, window_size):
    embeddings = FromPreTrained('embeddings.txt', 'words.txt')
    F2I = embeddings.get_word_to_idx()
    weights = embeddings.get_embeddings()
    train_data = DataReader(window_size, data_type=data_type, F2I=F2I, to_lower=True)
    L2I = train_data.get_l2i()
    I2L = train_data.get_i2l()
    I2F = train_data.get_i2f()
    return train_data, F2I, L2I, I2L, I2F, weights


def tagger_3(data_processor, data_type):
    batch_size = 1000
    hidden_size = 100
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 10
    prefix_size = 3
    suffix_size = 3
    train_data, F2I, L2I, I2L, I2F, weights = data_processor(data_type=data_type, window_size=window_size)
    dev_data = DataReader(window_size, data_type=data_type, mode="dev", F2I=F2I, L2I=L2I, to_lower=True)
    create_dictionaries(prefix_size, suffix_size, train_data.get_sentences(), I2F)
    output_size = len(L2I)
    vocab_size = len(F2I)
    prefix_vocab_size = len(PRE2I)
    suffix_vocab_size = len(SUF2I)
    model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size, prefix_vocab_size,
                  suffix_vocab_size, weights)
    model = trainer_loop(model, train_data.data_loader(batch_size),
                         dev_data.data_loader(batch_size), I2L, learning_rate, epochs)
    # test_parser = DataReader(window_size, data_type=data_type, mode='test', to_lower=True)
    # predict(model, test_parser.data_loader(shuffle=False), data_type, I2L)


def pre_trained_or_normal(arg):
    if arg == 'normal':
        return normal_loader
    elif arg == 'pre_trained':
        return pre_trained_loader
    return None


def is_data_type(arg):
    return arg == 'ner' or arg == 'pos'


def arguments_handler():
    args = sys.argv
    data_type = 'pos'
    data_processor = normal_loader
    if len(args) == 2:
        _data_processor = pre_trained_or_normal(args[1])
        if _data_processor:
            data_processor = _data_processor
        else:
            data_type = args[1]
    elif len(args) == 3:
        if is_data_type(args[1]):
            data_type = args[1]
            data_processor = pre_trained_or_normal(args[2])
        else:
            data_type = args[2]
            data_processor = pre_trained_or_normal(args[1])
    if is_data_type(data_type) and data_processor:
        tagger_3(data_processor, data_type)
    else:
        print("Invalid data type or function call;\n"
              "\tValid data types:\n"
              "\t\t1. ner\n"
              "\t\t2. pos\n"
              "\tValid function calls:\n"
              "\t\t1. normal\n"
              "\t\t2. pre_trained\n"
              "Order doesn't matter, default values are normal and pos")


if __name__ == "__main__":
    # Enter values by hand or accept arguments to main
    arguments_handler()
    # tagger_3(normal_loader, 'ner', 'normal')

