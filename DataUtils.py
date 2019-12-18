import re
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

START = "*START*"
END = "*END*"
UNKNOWN = '*UNKNOWN*'


def get_file_directory(data_name, data_kind):
    return "./data/{0}/{1}".format(data_name, data_kind)


class Parser:
    def __init__(self, window_size, data_name='pos', data_kind="train", F2I={}, L2I={}):
        file_dir = get_file_directory(data_name, data_kind)
        self.file = open(file_dir, 'r')
        self.windows = []
        self.window_labels = []
        self.window_size = window_size
        self.sentences = []
        self.word_labels = []
        self.F2I = F2I
        self.L2I = L2I
        self.is_pos = (data_name == 'pos')
        self.data_kind = data_kind

    def parse_to_indexed_windows(self, convert_digits=True, to_lower=True):
        self.parse_sentences(convert_digits=True, to_lower=True)
        self.create_windows_list_from_sentences()
        self.convert_sentences_windows_to_indexes()

    def create_windows_list_from_sentences(self):
        for sentence in self.sentences:
            if len(sentence) < self.window_size:
                raise ValueError("Sentences must be bigger then window size")
            current_sentence_window = list()
            # curr_sentence_label = ""
            last_element = len(sentence) - self.window_size + 1
            for i in range(last_element):
                curr_sentence = [word for word in sentence[i:i + self.window_size]]
                current_sentence_window.append(curr_sentence)
            self.windows.extend(current_sentence_window)

        if self.data_kind != "test":
            for label in self.word_labels:
                last_element = len(label) - self.window_size + 1
                for i in range(last_element):
                    curr_sentence_label = label[i + self.window_size // 2]
                    self.window_labels.append(curr_sentence_label)

    def convert_sentences_windows_to_indexes(self):
        f2i = self.get_f2i()
        l2i = self.get_l2i()
        for sentence in self.windows:
            for index, word in enumerate(sentence):
                if word in f2i:
                    sentence[index] = f2i[word]
                else:
                    sentence[index] = f2i[UNKNOWN]
        if self.data_kind != "test":
            for index, label in enumerate(self.window_labels):
                if label in l2i:
                    self.window_labels[index] = l2i[label]
                else:
                    self.window_labels[index] = l2i[UNKNOWN]

    def parse_sentences(self, convert_digits=True, to_lower=True):
        # parse by spaces if post, if ner parse by tab.
        delimiter = ' ' if self.is_pos else '\t'
        current_sentence_words = list()
        current_sentence_labels = list()
        for raw in self.file:
            raw_splitted = raw.split('\n')
            raw_splitted = raw_splitted[0].split(delimiter)
            word = raw_splitted[0]
            if word != '':
                # convert all chars to lower case.
                if to_lower:
                    word = word.lower()
                # if we want to convert each digit to be DG for similarity, '300' = '400'.
                if convert_digits:
                    word = re.sub('[0-9]', 'DG', word)
                label = raw_splitted[1]
                current_sentence_words.append(word)
                current_sentence_labels.append(label)
            else:
                full_sentence_words = [START, START] + current_sentence_words + [END, END]
                self.sentences.append(full_sentence_words)
                full_sentence_labels = [START, START] + current_sentence_labels + [END, END]
                self.word_labels.append(full_sentence_labels)

                current_sentence_words.clear()
                current_sentence_labels.clear()

    def get_sentences(self):
        return self.windows

    def get_labels(self):
        return self.window_labels

    def get_f2i(self):
        if not self.F2I:
            self.F2I = {f: i for i, f in
                        enumerate(list(sorted(set([w for sublist in self.sentences for w in sublist]))))}
            self.F2I[UNKNOWN] = len(self.F2I)
        return self.F2I

    def get_l2i(self):
        if not self.L2I:
            self.L2I = {l: i for i, l in
                        enumerate(list(sorted(set([w for sublist in self.word_labels for w in sublist]))))}
            self.L2I[UNKNOWN] = len(self.L2I)
        return self.L2I

    def get_i2f(self):
        i2f = {i: l for l, i in self.F2I.items()}
        return i2f

    def get_i2l(self):
        i2l = {i: l for l, i in self.L2I.items()}
        return i2l

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
    p.parse_sentences()
    p.create_windows_list_from_sentences()
    p.convert_sentences_windows_to_indexes()
