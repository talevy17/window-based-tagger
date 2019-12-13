

class Parser:
    def __init__(self, file):
       self.file = open(file, 'r')
       self.tup = []
       self.word_vector = []

    def parse_sentences(self):
        sentence = []
        data = []
        for raw in self.file:
            sentence.append('STARTT')
            sentence.append('STARTT')
            data.append(('STARTT', 'STARTT'))
            data.append(('STARTT', 'STARTT'))
            raw_splitted = raw.split('\n')
            raw_splitted = raw_splitted[0].split(' ')
            word = raw_splitted[0]
            if word == '':
                sentence.append('ENDD')
                sentence.append('ENDD')
                data.append(('ENDD', 'ENDD'))
                data.append(('ENDD', 'ENDD'))
                self.word_vector.append(sentence)
                self.tup.append(data)
                sentence = []
                data = []
                continue
            label = raw_splitted[1]
            sentence.append(word)
            data.append((word, label))

    def get_tuples(self):
        return self.tup

    def get_sentences(self):
        return self.word_vector

    def get_f2i(self):
        words = [sentence for sentence in self.tup]
        return {f: i for i, f in enumerate(list(sorted(set([w[0] for w in words]))))}

    def get_l2i(self):
        labels = [sentence for sentence in self.tup]
        return {l: i for i, l in enumerate(list(sorted(set([w[1] for w in labels]))))}


if __name__ == '__main__':
    p = Parser("./Dataset/pos/train")
    p.parse_sentences()








