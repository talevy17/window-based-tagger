import torch
import torch.nn as nn
import time
from parser import Parser
from Sentences import Sentences
from torch.utils.data import DataLoader
import numpy as np


batch_size = 1
hidden_size = 100
embedding_length = 50
window_size = 5
learning_rate = 0.01
epochs = 10


class Model(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, window_size):
        super(Model, self).__init__()
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

    def embed_and_concat(self, x):
        concat_vector = []
        for word in x:
            concat_vector.append(self.embedded(word))
        return torch.cat(tuple(concat_vector), 0)

    def forward(self, x):
        data = self.embed_and_concat(x)
        data = self.hidden(data)
        data = torch.tanh(data)
        data = self.out(data)
        return nn.functional.softmax(data.reshape(1, -1), dim=1)


def one_hot(label, size):
    y_hot = np.zeros(size)
    y_hot[label] = 1
    return torch.from_numpy(y_hot.reshape(-1, 1)).long()


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def get_accuracy(prediction, y):
    probs = torch.softmax(prediction, dim=1)
    winners = probs.argmax(dim=1)
    correct = (winners == y.argmax(dim=1)).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, loader, optimizer, loss_func, epoch):
    epoch_loss = 0
    epoch_acc = 0
    label_size = len(L2I)
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for batch in loader:
        optimizer.zero_grad()
        prediction = model(batch[0])
        label = one_hot(batch[1], label_size)
        loss = loss_func(prediction, label[0])
        epoch_acc += get_accuracy(prediction, label[0])
        epoch_loss += loss
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return float(epoch_loss) / len(loader), float(epoch_acc) / len(loader), model


def evaluate(model, loader, loss_func, epoch):
    epoch_loss = 0
    epoch_acc = 0
    label_size = len(L2I)
    model.eval()
    print(f'Epoch: {epoch + 1:02} | Starting Evaluation...')
    with torch.no_grad:
        for batch in loader:
            prediction = model(batch[0])
            label = one_hot(batch[1], label_size)
            loss = loss_func(prediction, label[0])
            epoch_acc += get_accuracy(prediction, label[0])
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_train = Parser('./data/train_5000')
    vocab_valid = Parser('./data/pos/dev')
    vocab_train.parse_sentences()
    vocab_valid.parse_sentences()
    L2I = vocab_train.get_l2i()
    F2I = vocab_train.get_f2i()
    vocab_valid.replace_non_vocab(F2I, L2I)
    output_size = len(L2I)
    vocab_size = len(F2I)
    model = Model(batch_size, output_size, hidden_size, vocab_size, embedding_length, window_size)
    model = model
    train_loader = Sentences(vocab_train.get_sentences(), F2I, L2I, window_size)
    valid_loader = Sentences(vocab_valid.get_sentences(), F2I, L2I, window_size)
    model = iterate_model(model, DataLoader(train_loader, batch_size=batch_size, shuffle=True),
                          DataLoader(valid_loader, batch_size=batch_size, shuffle=True))
