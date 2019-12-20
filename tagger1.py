
from Model import Model
from ModelTrainer import trainer_loop, predict
from DataUtils import DataReader
import sys


if __name__ == "__main__":
    batch_size = 1028
    hidden_size = 128
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 100
    data_type = sys.argv[1] if len(sys.argv) > 1 else 'pos'
    train_data = DataReader(window_size, data_type=data_type, to_lower=True)
    L2I = train_data.get_l2i()
    F2I = train_data.get_f2i()
    I2L = train_data.get_i2l()
    output_size = len(L2I)
    vocab_size = len(F2I)
    dev_data = DataReader(window_size, data_type=data_type, mode="dev", F2I=F2I, L2I=L2I, to_lower=True)
    model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size)
    model = trainer_loop(model, train_data.data_loader(batch_size),
                         dev_data.data_loader(batch_size), I2L, learning_rate, epochs)
    test_parser = DataReader(window_size, data_type=data_type, mode='test', to_lower=True)
    predict(model, test_parser.data_loader(shuffle=False), 'ner', I2L)
