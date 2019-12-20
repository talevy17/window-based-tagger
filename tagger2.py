from Parser import Parser, UNKNOWN
from top_k import PreTrainedLoader
from tagger1 import make_loader, iterate_model
from utils import make_test_loader, predict_by_windows
import torch
import torch.nn as nn
import numpy as np
import sys

batch_size = 1000
hidden_size = 100
embedding_length = 50
window_size = 5
learning_rate = 0.01
epochs = 1


class Model(nn.Module):
	def __init__(self, output_size, hidden_size, embedding_length, window_size, weights):
		super(Model, self).__init__()
		self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=False)
		self.concat_size = window_size * embedding_length
		self.hidden = nn.Linear(self.concat_size, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		data = self.embed(x).view(-1, self.concat_size)
		data = self.hidden(data)
		data = torch.tanh(data)
		data = self.out(data)
		return self.softmax(data)


def tagger_2():
	data_name = sys.argv[1]
	
	pretrained = PreTrainedLoader('./data/pretrained/wordVectors.txt', './data/pretrained/vocab.txt')
	F2I = pretrained.get_dict()
	weights = pretrained.get_weights()
	weights = np.concatenate((weights, np.zeros((1, embedding_length))))
	vocab_train = Parser(window_size, data_name=data_name, F2I=F2I)
	vocab_train.parse_to_indexed_windows()
	L2I = vocab_train.get_l2i()
	I2L = vocab_train.get_i2l()
	vocab_valid = Parser(window_size, data_name=data_name, data_kind="dev", F2I=F2I, L2I=L2I)
	vocab_valid.parse_to_indexed_windows()
	output_size = len(L2I)
	model = Model(output_size, hidden_size, embedding_length, window_size, weights)
	iterate_model(model, make_loader(vocab_train, batch_size), make_loader(vocab_valid, batch_size), I2L)

	test_parser = Parser(window_size, data_name, 'test')
	test_parser.parse_to_indexed_windows()
	predict_by_windows(model, make_test_loader(test_parser), data_name, I2L)


if __name__ == "__main__":
	tagger_2()
