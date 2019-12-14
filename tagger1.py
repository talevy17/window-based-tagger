import torch
import torch.nn as nn
import time
from parser import Parser
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


batch_size = 100
hidden_size = 100
embedding_length = 50
window_size = 5
learning_rate = 0.01
epochs = 10


class Model(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, embedding_length, window_size):
        super(Model, self).__init__()
        torch.manual_seed(3)
        self.embed = nn.Embedding(vocab_size, embedding_length)
        self.concat_size = window_size * embedding_length
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)
        self.hidden = nn.Linear(self.concat_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        data = self.embed(x).view(-1, self.concat_size)
        data = self.hidden(data)
        data = torch.tanh(data)
        data = self.out(data)
        return self.softmax(data)


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def get_accuracy(prediction, y):
    acc = 0
    for pred, label in zip(prediction, y):
        if pred.argmax() == label:
            acc += 1
    return acc / len(y)


def train(model, loader, optimizer, loss_func, epoch):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for x, y in loader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_func(prediction, y)
        epoch_acc += get_accuracy(prediction, y)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return float(epoch_loss) / len(loader), float(epoch_acc) / len(loader), model


def evaluate(model, loader, loss_func, epoch):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    print(f'Epoch: {epoch + 1:02} | Starting Evaluation...')
    for x, y in loader:
        prediction = model(x)
        loss = loss_func(prediction, y)
        epoch_acc += get_accuracy(prediction, y)
        epoch_loss += loss
    print(f'Epoch: {epoch + 1:02} | Finished Evaluation')
    return float(epoch_loss) / len(loader), float(epoch_acc) / len(loader)


def iterate_model(model, train_loader, validation_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc, model = train(model, train_loader, optimizer, loss, epoch)
        val_loss, val_acc = evaluate(model, validation_loader, loss, epoch)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')
    return model


def make_loader(parser):
    x, y = parser.get_sentences(), parser.get_labels()
    x, y = torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))
    x, y = x.type(torch.long), y.type(torch.long)
    return DataLoader(TensorDataset(x, y), batch_size, shuffle=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_train = Parser('./data/pos/train', window_size)
    vocab_train.parse_sentences()
    L2I = vocab_train.get_l2i()
    F2I = vocab_train.get_f2i()
    vocab_valid = Parser('./data/pos/dev', window_size, F2I, L2I)
    vocab_valid.parse_sentences()
    output_size = len(L2I)
    vocab_size = len(F2I)
    model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size)
    model = model
    model = iterate_model(model, make_loader(vocab_train), make_loader(vocab_valid))
