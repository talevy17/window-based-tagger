import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length,
                 freeze_embeddings=False):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        torch.manual_seed(3)
        self.embedded = nn.Embedding(vocab_size, embedding_length)
        nn.init.uniform_(self.embedded.weight, -1.0, 1.0)
        # self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(self.embedded.weight), freeze=freeze_embeddings)
        self.hidden = nn.Linear(5 * embedding_length, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        data = torch.cat((self.embedded(x[0]), self.embedded(x[1]),
                          self.embedded(x[2]), self.embedded(x[3]), self.embedded(x[4])), 0)
        data = self.hidden(data)
        data = nn.functional.tanh(data)
        data = self.out(data)
        return nn.functional.softmax(data)

