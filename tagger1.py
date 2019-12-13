import torch
import torch.nn as nn
import time


batch_size = 8
hidden_size = 100
embedding_length = 50
window_size = 5
learning_rate = 0.01
epochs = 10


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

    def embed_and_concat(self, x):
        concat_vector = []
        for word in x:
            concat_vector.append(self.embedded(self.F2I[word]))
        return torch.cat(tuple(concat_vector), 0)

    def forward(self, x):
        data = self.embed_and_concat(x)
        data = self.hidden(data)
        data = nn.functional.tanh(data)
        data = self.out(data)
        return nn.functional.softmax(data)


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def get_accuracy(prediction, y):
    winners = prediction.argmax(dim=1)
    return winners == y.argmax(dim=1)  # convert into float for division


def train_sentence(sentence, model, optimizer, loss_func):
    acc = 0
    ep_loss = 0
    for w1, w2, w3, w4, w5 in zip(sentence[:-4], sentence[1:-3], sentence[2:-2], sentence[3:-1], sentence[4:]):
        optimizer.zero_grad()
        prediction = model([w1[0], w2[0], w3[0], w4[0], w5[0]])
        loss = loss_func(prediction, w3[1])
        acc += get_accuracy(prediction, w3[1])
        ep_loss += loss
        loss.backward()
        optimizer.step()
    return acc, len(sentence) - 4, ep_loss


def train(model, train_set, optimizer, loss_func, epoch):
    epoch_loss = 0
    epoch_acc = 0
    words = 0
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for index, batch in enumerate(train_set):
        acc, num_words, loss = train_sentence(batch[0], model, optimizer, loss_func)
        epoch_loss += loss
        epoch_acc += acc
        words += num_words
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return epoch_loss / words, epoch_acc / words


def evaluate_sentence(sentence, model, optimizer, loss_func):
    acc = 0
    ep_loss = 0
    with torch.no_grad():
        for w1, w2, w3, w4, w5 in zip(sentence[:-4], sentence[1:-3], sentence[2:-2], sentence[3:-1], sentence[4:]):
            prediction = model([w1[0], w2[0], w3[0], w4[0], w5[0]])
            loss = loss_func(prediction, w3[1])
            acc += get_accuracy(prediction, w3[1])
            ep_loss += loss
        return acc, len(sentence) - 4, ep_loss


def evaluate(model, train_set, optimizer, loss_func, epoch):
    epoch_loss = 0
    epoch_acc = 0
    words = 0
    model.eval()
    print(f'Epoch: {epoch + 1:02} | Starting Evaluating...')
    for index, batch in enumerate(train_set):
        acc, num_words, loss = evaluate_sentence(batch[0], model, optimizer, loss_func)
        epoch_loss += loss
        epoch_acc += acc
        words += num_words
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return epoch_loss / words, epoch_acc / words


def iterate_model(model, train_set, validation_set):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_set, optimizer, loss, epoch)
        val_loss, val_acc = evaluate(model, validation_set, loss, epoch)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')


if __name__ == "__main__":
    output_size = len(L2I)
    vocab_size = len(vocab)
    model = Model(batch_size, output_size, hidden_size, vocab_size, embedding_length, window_size, F2I)
    iterate_model(model, train_set, validation_set)
