import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

START = "*START*"
END = "*END*"
UNKNOWN = '*UNKNOWN*'


class Parser:
    def __init__(self, window_size, data_name='pos', data_kind="train", F2I={}, L2I={}, to_lower=True):
        with open("./Data/{0}/{1}".format(data_name, data_kind), 'r') as file:
            data = file.readlines()
        self.windows = []
        self.window_labels = []
        sentences, labels = self.parse_sentences(data, data_name == 'pos', to_lower)
        self.F2I = F2I if F2I else self.create_dict(sentences)
        self.L2I = L2I if L2I else self.create_dict(labels)
        self.create_windows_list_from_sentences(sentences, labels, window_size, data_kind)
        self.convert_sentences_windows_to_indexes(data_kind)

    def create_windows_list_from_sentences(self, sentences, labels, window_size, data_kind):
        for sentence in sentences:
            if len(sentence) < window_size:
                raise ValueError("Sentences must be bigger then window size")
            current_sentence_window = []
            last_element = len(sentence) - window_size + 1
            for i in range(last_element):
                curr_sentence = [word for word in sentence[i:i + window_size]]
                current_sentence_window.append(curr_sentence)
            self.windows.extend(current_sentence_window)

        if data_kind != "test":
            for label in labels:
                last_element = len(label) - window_size + 1
                for i in range(last_element):
                    curr_sentence_label = label[i + window_size // 2]
                    self.window_labels.append(curr_sentence_label)

    def convert_sentences_windows_to_indexes(self, data_kind):
        f2i = self.get_f2i()
        l2i = self.get_l2i()
        for sentence in self.windows:
            for index, word in enumerate(sentence):
                if word in f2i:
                    sentence[index] = f2i[word]
                else:
                    sentence[index] = f2i[UNKNOWN]
        if data_kind != "test":
            for index, label in enumerate(self.window_labels):
                if label in l2i:
                    self.window_labels[index] = l2i[label]
                else:
                    self.window_labels[index] = l2i[UNKNOWN]

    @staticmethod
    def parse_sentences(data, is_pos, to_lower):
        # parse by spaces if post, if ner parse by tab.
        delimiter = ' ' if is_pos else '\t'
        current_sentence = []
        current_labels = []
        sentences = []
        labels = []
        for row in data:
            row_spitted = row.split('\n')
            row_spitted = row_spitted[0].split(delimiter)
            word = row_spitted[0]
            if word != '':
                # convert all chars to lower case.
                if to_lower:
                    word = word.lower()
                label = row_spitted[1]
                current_sentence.append(word)
                current_labels.append(label)
            else:
                full_sentence_words = [START, START] + current_sentence + [END, END]
                sentences.append(full_sentence_words)
                full_sentence_labels = [START, START] + current_labels + [END, END]
                labels.append(full_sentence_labels)
                current_sentence.clear()
                current_labels.clear()
        return sentences, labels

    def get_sentences(self):
        return self.windows

    def get_labels(self):
        return self.window_labels

    def get_f2i(self):
        return self.F2I

    def get_l2i(self):
        return self.L2I

    @staticmethod
    def create_dict(data):
        data_dict = {f: i for i, f in enumerate(list(sorted(set([w for row in data for w in row]))))}
        data_dict[UNKNOWN] = len(data_dict)
        return data_dict

    def get_i2f(self):
        return {i: l for l, i in self.F2I.items()}

    def get_i2l(self):
        return {i: l for l, i in self.L2I.items()}

    def data_loader(self, batch_size=1, shuffle=True):
        windows, labels = torch.from_numpy(np.array(self.windows)), torch.from_numpy(np.array(self.window_labels))
        windows, labels = windows.type(torch.long), labels.type(torch.long)
        return DataLoader(TensorDataset(windows, labels), batch_size, shuffle=shuffle)


class FromPreTrained:
    def __init__(self, vectors, vocab):
        word_vectors = np.loadtxt(vectors)
        self.embeddings = np.concatenate((word_vectors, np.zeros((1, len(word_vectors[0])))))
        file = open(vocab, 'r')
        self.corpus = {f.split('\n')[0]: i for i, f in enumerate(file)}
        file.close()
        self.corpus[UNKNOWN] = len(self.corpus)
        self.idx_to_word = {i: f for f, i in self.corpus.items()}

    def get_embeddings(self):
        return self.embeddings

    def get_word_to_idx(self):
        return self.corpus

    def get_idx_to_word(self):
        return self.idx_to_word


if __name__ == '__main__':
    p = Parser(window_size=5, data_name='ner')

