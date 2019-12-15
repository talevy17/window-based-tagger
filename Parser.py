import re


class Parser:
    def __init__(self, file, window_size, F2I={}, L2I={}):
        self.file = open(file, 'r')
        self.window_sentences = []
        self.window_sentences_labels = []
        self.window_size = window_size
        self.F2I = F2I
        self.L2I = L2I

    def create_window_list_from_sentence(self, sentence_list):
        window_sentences = list()
        window_sentences_labels = list()
        if len(sentence_list) < self.window_size:
            raise ValueError("Sentences must be bigger then window size")
        last_element = len(sentence_list) - self.window_size + 1
        for i in range(last_element):
            curr_sentence = [tup[0] for tup in sentence_list[i:i + self.window_size]]
            window_sentences.append(curr_sentence)
            curr_sentence_label = sentence_list[i + self.window_size // 2][1]
            window_sentences_labels.append(curr_sentence_label)
        return window_sentences, window_sentences_labels

    def create_window_test_list_from_sentence(self, sentence_list):
        window_sentences = list()
        if len(sentence_list) < self.window_size:
            raise ValueError("Sentences must be bigger then window size")
        last_element = len(sentence_list) - self.window_size + 1
        for i in range(last_element):
            curr_sentence = [word for word in sentence_list[i:i + self.window_size]]
            window_sentences.append(curr_sentence)
        return window_sentences

    def convert_sentences_to_indexes(self):
        f2i = self.get_f2i()
        l2i = self.get_l2i()
        for sentence in self.window_sentences:
            for index, word in enumerate(sentence):
                if word in f2i:
                    sentence[index] = f2i[word]
                else:
                    sentence[index] = f2i['']
        for index, label in enumerate(self.window_sentences_labels):
            if label in l2i:
                self.window_sentences_labels[index] = l2i[label]
            else:
                self.window_sentences_labels[index] = l2i['']

    def parse_sentences(self, delimeter, convert_digits=True):
        current_sentence = list()
        for raw in self.file:
            raw_splitted = raw.split('\n')
            raw_splitted = raw_splitted[0].split(delimeter)
            word = raw_splitted[0]
            if word != '':
                # convert all chars to lower case.
                word = word.lower()
                # if we want to convert each digit to be DG for similarity, '300' = '400'.
                if convert_digits:
                    word = re.sub('[0-9]', 'DG', word)
                label = raw_splitted[1]
                current_sentence.append((word, label))
            else:
                full_sentence = [('STARTT', 'STARTT'), ('STARTT', 'STARTT')] + current_sentence + [('ENDD', 'ENDD'),
                                                                                                   ('ENDD', 'ENDD')]
                sentences, sentences_labels = self.create_window_list_from_sentence(full_sentence)
                self.window_sentences.extend(sentences)
                self.window_sentences_labels.extend(sentences_labels)
                current_sentence.clear()

        # convert words to indexes
        self.convert_sentences_to_indexes()

    def parse_test_sentences(self, convert_digits=True):
        current_sentence = list()
        for raw in self.file:
            raw_splitted = raw.split('\n')
            word = raw_splitted[0]
            if word != '':
                # convert all chars to lower case.
                word = word.lower()
                # if we want to convert each digit to be DG for similarity, '300' = '400'.
                if convert_digits:
                    word = re.sub('[0-9]', 'DG', word)
                current_sentence.append(word)
            else:
                full_sentence = ['STARTT', 'STARTT'] + current_sentence + ['ENDD', 'ENDD']
                sentences = self.create_window_test_list_from_sentence(full_sentence)
                self.window_sentences.extend(sentences)
                current_sentence.clear()

        # convert words to indexes
        self.convert_sentences_to_indexes()

    def get_sentences(self):
        return self.window_sentences

    def get_labels(self):
        return self.window_sentences_labels

    def get_f2i(self):
        if not self.F2I:
            self.F2I = {f: i for i, f in
                        enumerate(list(sorted(set([w for sublist in self.window_sentences for w in sublist]))))}
            self.F2I[''] = len(self.F2I)
        return self.F2I

    def get_l2i(self):
        if not self.L2I:
            self.L2I = {l: i for i, l in enumerate(list(sorted(set([w for w in self.window_sentences_labels]))))}
            self.L2I[''] = len(self.L2I)
        return self.L2I

    def get_i2l(self):
        i2l = {i: l for l, i in self.L2I.items()}
        i2l[len(i2l)] = ''
        return i2l


if __name__ == '__main__':
    p = Parser("./Dataset/pos/test", window_size=5)
    p.parse_test_sentences(convert_digits=True)
