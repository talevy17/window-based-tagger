
import numpy as np
from DataUtils import FromPreTrained


def cosine_distance(a, b):
    return np.divide(np.dot(a, b), (np.linalg.norm(a) * np.linalg.norm(b)))


def get_k_nearest(k, anchor, weights, word_to_idx):
    results = []
    embedded_anchor = weights[word_to_idx[anchor]]
    for word, weight in zip(word_to_idx.keys(), weights):
        if not word == anchor:
            results.append((cosine_distance(embedded_anchor, weight), word))
    results.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in results[:k]]


def top_k():
    embed = FromPreTrained('./Data/pretrained/embeddings.txt', './Data/pretrained/words.txt')
    words = ['dog', 'england', 'john', 'explode', 'office']
    weights = embed.get_embeddings()
    word_to_idx = embed.get_word_to_idx()
    for word in words:
        print(get_k_nearest(5, word, weights, word_to_idx))


if __name__ == "__main__":
    top_k()
