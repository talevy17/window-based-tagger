

class Parser:
    def __init__(self, file):
       self.file = open(file, 'r')
       self.tup = []
       self.word_vector = set()

    def parse_sentences(self):
        data = []
        self.word_vector.add(('STARTT', 'STARTT'))
        self.word_vector.add(('ENDD', 'ENDD'))
        data.append(('STARTT', 'STARTT'))
        data.append(('STARTT', 'STARTT'))
        for raw in self.file:
            raw_splitted = raw.split('\n')
            raw_splitted = raw_splitted[0].split(' ')
            word = raw_splitted[0]
            if word == '':
                data.append(('ENDD', 'ENDD'))
                data.append(('ENDD', 'ENDD'))
                self.tup.append(data)
                data = []
                data.append(('STARTT', 'STARTT'))
                data.append(('STARTT', 'STARTT'))
                continue
            label = raw_splitted[1]
            self.word_vector.add((word, label))
            data.append((word, label))

    def get_tuples(self):
        return self.tup

    def replace_non_vocab(self, vocab):
        for sentence in self.tup:
            for i, w in enumerate(sentence):
                if not w[0] in vocab:
                    sentence[i] = ('', '')

    def get_sentences(self):
        return self.word_vector

    def get_f2i(self):
        F2I = {f: i for i, f in enumerate(list(sorted(set([w[0] for w in self.word_vector]))))}
        F2I[''] = len(F2I)
        return F2I

    def get_l2i(self):
        L2I = {l: i for i, l in enumerate(list(sorted(set([w[1] for w in self.word_vector]))))}
        L2I[''] = len(L2I)
        return L2I


if __name__ == '__main__':
    p = Parser("./Dataset/pos/train")
    p.parse_sentences()








