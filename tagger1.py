import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, window_size, F2I):
        super(Model, self).__init__()
        self.F2I = F2I
        self.window_size = window_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        torch.manual_seed(3)
        self.embedded = nn.Embedding(vocab_size, embedding_length)
        nn.init.uniform_(self.embedded.weight, -1.0, 1.0)
        self.hidden = nn.Linear(window_size * embedding_length, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def concat(self, x):
        concat_vector = []
        for word in x:
            concat_vector.append(self.embedded(self.F2I[word]))
        return tuple(concat_vector)

    def forward(self, x):
        data = torch.cat(self.concat(x), 0)
        data = self.hidden(data)
        data = nn.functional.tanh(data)
        data = self.out(data)
        return nn.functional.softmax(data)

