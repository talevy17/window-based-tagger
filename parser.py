

class parser:
    def __init__(self, file):
       self.file = open(file, 'r')
       self.dictionary ={}
       self.dictionary['STARTT'] ='STARTT'
       self.dictionary['ENDD'] ='ENDD'
       self.word_vector = []

    def parse_sentences(self):
        sentence = []
        for raw in self.file:
            sentence.append('STARTT')
            sentence.append('STARTT')
            raw_splitted = raw.split('\n')
            raw_splitted = raw_splitted[0].split(' ')
            word = raw_splitted[0]
            if word == '':
                sentence.append('ENDD')
                sentence.append('ENDD')
                self.word_vector.append(sentence)
                sentence= []
                continue
            label = raw_splitted[1]
            sentence.append(word)
            self.dictionary[word] = label

if __name__ == '__main__':
    p = parser("./Dataset/pos/train")
    p.parse_sentences()








