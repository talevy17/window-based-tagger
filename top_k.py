import numpy as np
from numpy import dot
from numpy.linalg import norm


class PreTrainedLoader:
    def __init__(self, vectors, vocab):
        self.vectors = np.loadtxt(vectors)
        file = open(vocab, 'r')
        self.vocab = {f.split('\n')[0]: i for i, f in enumerate(file)}
        file.close()
        self.i2f = {i: f for f, i in self.vocab.items()}

    def get_weights(self):
        return self.vectors

    def get_dict(self):
        return self.vocab

    def get_i2f(self):
        return self.i2f


if __name__ == "__main__":
    loader = PreTrainedLoader('./Data/pretrained/wordVectors.txt', './Data/pretrained/vocab.txt')
    words = ['dog', 'england', 'john', 'explode', 'office']
    weights = loader.get_weights()
    f2i = loader.get_dict()
    i2f = loader.get_i2f()
    for word in words:
        results = np.zeros(len(f2i))
        word_index = f2i[word]
        word_vector = weights[word_index]
        for index, vector in enumerate(weights):
            if not index == word_index:
                results[index] = dot(word_vector, vector) / norm(word_vector) * norm(vector)
            else:
                results[index] = np.inf
        similars = []
        for i in range(5):
            index = np.argmin(results)
            similars.append(i2f[index])
            results[index] = np.inf
        print(similars)
