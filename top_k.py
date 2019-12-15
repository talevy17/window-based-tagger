import numpy as np
from numpy import dot
from numpy.linalg import norm
import re


class PreTrainedLoader:
    def __init__(self, vectors, vocab, convert_digits=True):
        self.vectors = np.loadtxt(vectors)
        file = open(vocab, 'r')
        if convert_digits:
            self.vocab = {re.sub('[0-9]', 'DG', f.split('\n')[0]): i for i, f in enumerate(file)}
        else:
            self.vocab = {f.split('\n')[0]: i for i, f in enumerate(file)}
        file.close()
        self.i2f = {i: f for f, i in self.vocab.items()}

    def get_weights(self):
        return self.vectors

    def get_dict(self):
        return self.vocab

    def get_i2f(self):
        return self.i2f


def sim(vector_a, vector_b):
    return dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b))


def top(k, key, weights, f2i):
    results = []
    word_vector = weights[f2i[key]]
    for word, vector in zip(f2i.keys(), weights):
        if not word == key:
            results.append((sim(word_vector, vector), word))
    results.sort(key=lambda x: x[0], reverse=True)
    return [element[1] for element in results[:k]]


if __name__ == "__main__":
    loader = PreTrainedLoader('./Data/pretrained/wordVectors.txt', './Data/pretrained/vocab.txt')
    words = ['dog', 'england', 'john', 'explode', 'office']
    weights = loader.get_weights()
    f2i = loader.get_dict()
    i2f = loader.get_i2f()
    for word in words:
        print(top(5, word, weights, f2i))
