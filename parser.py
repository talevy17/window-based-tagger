

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

    def get_sentences(self):
        return self.word_vector

    def get_f2i(self):
        return {f: i for i, f in enumerate(list(sorted([w[0] for w in self.word_vector])))}

    def get_l2i(self):
        return {l: i for i, l in enumerate(list(sorted([w[1] for w in self.word_vector])))}


if __name__ == '__main__':
    p = Parser("./Dataset/pos/train")
    p.parse_sentences()








