import csv

import torch
import torch.nn as nn
import time
from Parser import Parser
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from utils import predict_by_windows


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
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        data = self.embed(x)
        data = torch.flatten(data, start_dim=1)
        data = self.hidden(data)
        data = torch.tanh(data)
        data = self.dropout(data)
        data = self.out(data)
        return self.softmax(data)


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def get_accuracy(prediction, y, I2L):
    acc = 0
    size = float(len(y))
    for pred, label in zip(prediction, y):
        if pred.argmax() == label:
            if not I2L[int(label)] == 'O':
                acc += 1
            else:
                size -= 1
    if size == 0:
        return 0
    return acc / size


def train(model, loader, optimizer, loss_func, epoch, I2L):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for x, y in loader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_func(prediction, y)
        epoch_acc += get_accuracy(prediction, y, I2L)
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return float(epoch_loss) / len(loader), float(epoch_acc) / len(loader), model


def evaluate(model, loader, loss_func, epoch, I2L):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    print(f'Epoch: {epoch + 1:02} | Starting Evaluation...')
    for x, y in loader:
        prediction = model(x)
        loss = loss_func(prediction, y)
        epoch_acc += get_accuracy(prediction, y, I2L)
        epoch_loss += loss
    print(f'Epoch: {epoch + 1:02} | Finished Evaluation')
    return float(epoch_loss) / len(loader), float(epoch_acc) / len(loader)


def iterate_model(model, train_loader, validation_loader, learning_rate, epochs, I2L):
    with open('tagger1_epochs_accuracy_ner.csv', mode='w') as file:
        fieldnames = ['Epoch Number', 'Train Loss', 'Train Accuracy', 'Val. Loss', 'Val Accuracy']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_acc, model = train(model, train_loader, optimizer, loss, epoch, I2L)
            val_loss, val_acc = evaluate(model, validation_loader, loss, epoch, I2L)
            end_time = time.time()
            epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')
            writer.writerow({'Epoch Number': str(epoch + 1), 'Train Loss': str(train_loss),
                             'Train Accuracy': str(train_acc * 100), 'Val. Loss': str(val_loss),
                             'Val Accuracy': str(val_acc * 100)})
    file.close()
    return model


def make_loader(parser, batch_size):
    x, y = parser.get_sentences(), parser.get_labels()
    x, y = torch.from_numpy(np.array(x)), torch.from_numpy(np.array(y))
    x, y = x.type(torch.long), y.type(torch.long)
    return DataLoader(TensorDataset(x, y), batch_size, shuffle=True)


def make_test_loader(parser):
    x = parser.get_sentences()
    x = torch.from_numpy(np.array(x))
    x = x.type(torch.long)
    return DataLoader(TensorDataset(x), 1, shuffle=False)

def tagger_1():
    batch_size = 2000
    hidden_size = 100
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_train = Parser(window_size, data_name="ner")
    vocab_train.parse_to_indexed_windows()
    L2I = vocab_train.get_l2i()
    F2I = vocab_train.get_f2i()
    I2L = vocab_train.get_i2l()
    vocab_valid = Parser(window_size, 'ner', "dev", F2I, L2I)
    vocab_valid.parse_to_indexed_windows()
    output_size = len(L2I)
    vocab_size = len(F2I)
    model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size)
    model = model
    model = iterate_model(model, make_loader(vocab_train, batch_size),
                          make_loader(vocab_valid, batch_size), learning_rate, epochs, I2L)
    test_parser = Parser(window_size, 'ner', 'test')
    test_parser.parse_to_indexed_windows()
    predict_by_windows(model, make_test_loader(test_parser), 'ner',I2L)


if __name__ == "__main__":
    tagger_1()
