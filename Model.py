import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, embedding_length, window_size,
                 weights=None, embedding_freeze=False):
        super(Model, self).__init__()
        if weights.any():
            self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=embedding_freeze)
        else:
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
