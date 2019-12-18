from Parser import Parser
from top_k import PreTrainedLoader
import tagger1, tagger2, tagger3
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def predict_by_windows(model, windows):
	predictions = list()
	for input in windows:
		y = tr.argmax(model(Variable(tr.LongTensor(input))).data)
		predictions.append(y)
	return predictions
