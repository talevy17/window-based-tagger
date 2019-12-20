from DataUtils import DataReader, FromPreTrained
from ModelTrainer import trainer_loop
from Model import Model
import sys


def tagger2(data_type):
    batch_size = 1000
    hidden_size = 100
    embedding_length = 50
    window_size = 5
    learning_rate = 0.01
    epochs = 10
    embeddings = FromPreTrained('embeddings.txt', 'words.txt')
    word_to_idx = embeddings.get_word_to_idx()
    weights = embeddings.get_embeddings()
    train_data = DataReader(window_size, data_type=data_type, F2I=word_to_idx, to_lower=True)
    L2I = train_data.get_l2i()
    I2L = train_data.get_i2l()
    dev_data = DataReader(window_size, data_type=data_type, mode="dev", F2I=word_to_idx, L2I=L2I, to_lower=True)
    output_size = len(L2I)
    vocab_size = len(word_to_idx)
    model = Model(output_size, hidden_size, vocab_size, embedding_length, window_size, weights)
    model = trainer_loop(model, train_data.data_loader(batch_size),
                         dev_data.data_loader(batch_size), I2L, learning_rate, epochs)


def arguments_handler():
    data_type = sys.argv[1] if len(sys.argv) > 1 else 'pos'
    if data_type == 'ner' or data_type == 'pos':
        tagger2(data_type)
    else:
        print("Invalid data type;\n"
              "\tValid data types are:\n"
              "\t\t1. pos\n"
              "\t\t2. ner\n"
              "default is pos")


if __name__ == "__main__":
    # you can accept arguments or choose a data type by hand
    # arguments_handler()
    tagger2('pos')
