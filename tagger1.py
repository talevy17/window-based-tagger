
from Model import Model
from ModelTrainer import trainer_loop, predict
from DataUtils import DataReader
import sys

STUDENT = {'name': 'Tal Levy, Lidor Alis',
           'ID': '---, ---'}


def tagger_1(data_type):
    batch_size = 1028
    hidden_size = 128
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 15
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
    predict(model, test_parser.data_loader(shuffle=False), data_type, I2L)


def arguments_handler():
    data_type = sys.argv[1] if len(sys.argv) > 1 else 'pos'
    if data_type == 'ner' or data_type == 'pos':
        tagger_1(data_type)
    else:
        print("Invalid data type;\n"
              "\tValid data types are:\n"
              "\t\t1. pos\n"
              "\t\t2. ner\n"
              "default is pos")


if __name__ == "__main__":
    # you can accept arguments or choose a data type by hand
    arguments_handler()
    # tagger1('ner')
    # tagger1('pos')
