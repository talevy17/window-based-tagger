import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Model import Model
from ModelTrainer import trainer_loop
from DataUtils import DataReader
import numpy as np

def make_test_loader(parser):
    x = parser.get_sentences()
    x = torch.from_numpy(np.array(x))
    x = x.type(torch.long)
    return DataLoader(TensorDataset(x), 1, shuffle=False)

def predict_by_windows(model, windows, file_type, L2I):
    with open('./data/test_{0}.txt'.format(file_type), mode='w') as file:
        predictions = list()
        for window in windows:
            y = model(window[0])
            _, y = torch.max(y, 1)
            y = L2I[int(y)]
            predictions.append(y)
            file.write("{0}\n".format(y))
    file.close()
    return predictions

def save_model_to_path(model, path="./data/model"):
	torch.save(model.state_dict(), path)


def load_model_from_path(model, path="./data/model"):
	model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    batch_size = 1000
    hidden_size = 100
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 1
    train_data = DataReader(window_size, data_name='ner')
    label_to_idx = train_data.get_l2i()
    word_to_idx = train_data.get_f2i()
    idx_to_label = train_data.get_i2l()
    dev_data = DataReader(window_size, 'ner', "dev", word_to_idx, label_to_idx)
    output_size = len(label_to_idx)
    vocab_size = len(word_to_idx)
    model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size)
    model = trainer_loop(model, train_data.data_loader(batch_size),
                         dev_data.data_loader(batch_size), idx_to_label, learning_rate, epochs)
    test_parser = DataReader(window_size, 'ner', 'test')
    predict_by_windows(model, make_test_loader(test_parser), 'ner', idx_to_label)
