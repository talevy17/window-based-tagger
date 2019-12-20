
from Model import Model
from ModelTrainer import trainer_loop, predict
from DataUtils import DataReader


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
    predict(model, test_parser.data_loader(shuffle=False), 'ner', idx_to_label)
