from torch.autograd import Variable

from Parser import Parser
from top_k import PreTrainedLoader
from tagger1 import iterate_model, make_loader
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
from utils import predict_by_windows, make_test_loader

prefixes = []
suffixes = []
PRE2I = {}
SUF2I = {}
batch_size = 100
hidden_size = 100
embedding_length = 50
window_size = 5
learning_rate = 0.01
epochs = 10
prefix_size = 3
suffix_size = 3


class Model(nn.Module):
	def __init__(self, output_size, hidden_size, vocab_size, embedding_length, window_size, prefix_size, suffix_size,
	             pre_train=False):
		super(Model, self).__init__()
		torch.manual_seed(3)
		if pre_train:
			pre_trained = PreTrainedLoader('./data/pretrained/wordVectors.txt', './Data/pretrained/vocab.txt')
			weights = pre_trained.get_weights()
			weights = np.concatenate((weights, np.zeros((1, embedding_length))))
			self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
		else:
			self.embed = nn.Embedding(vocab_size, embedding_length)
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

		data = (prefix_embed + normal_embed + suffix_embed).view(-1, self.concat_size)

		data = self.hidden(data)
		data = torch.tanh(data)
		data = self.out(data)
		return self.softmax(data)


def get_prefix_index_by_word_index(index):
	return PRE2I[prefixes[index]]


def get_suffix_index_by_word_index(index):
	return SUF2I[suffixes[index]]


def create_pre_suff_dicts(prefix_size, suffix_size, window_sentences, i2f):
	global PRE2I, SUF2I, prefixes, suffixes
	words = [word for sublist in window_sentences for word in sublist]
	prefixes = [i2f[word][: prefix_size] for word in words]
	suffixes = [i2f[word][-suffix_size:] for word in words]
	PRE2I = {pre: i for i, pre in enumerate(sorted(set(prefixes)))}
	SUF2I = {suf: i for i, suf in enumerate(sorted(set(suffixes)))}


def tagger_3():

	data_name = sys.argv[1]

	pre_train = len(sys.argv) > 1 and sys.argv[1] == "pre_train"
	vocab_train = Parser(window_size, data_name=data_name)
	vocab_train.parse_to_indexed_windows()
	L2I = vocab_train.get_l2i()
	F2I = vocab_train.get_f2i()
	I2L = vocab_train.get_i2l()
	I2F = vocab_train.get_i2f()

	vocab_valid = Parser(window_size, data_name, "dev", F2I, L2I)
	vocab_valid.parse_to_indexed_windows()
	create_pre_suff_dicts(prefix_size, suffix_size, vocab_valid.get_sentences(), I2F)
	output_size = len(L2I)
	vocab_size = len(F2I)
	prefix_vocab_size = len(PRE2I)
	suffix_vocab_size = len(SUF2I)
	model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size, prefix_vocab_size,
	              suffix_vocab_size, pre_train=pre_train)
	model = iterate_model(model, make_loader(vocab_train, batch_size), make_loader(vocab_valid, batch_size),
				I2L, epochs, batch_size, hidden_size, learning_rate)

	test_parser = Parser(window_size, data_name, 'test')
	test_parser.parse_to_indexed_windows()
	predict_by_windows(model, make_test_loader(test_parser), data_name, I2L)


if __name__ == "__main__":
	tagger_3()
