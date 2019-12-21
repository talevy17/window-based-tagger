
import numpy as np
from DataUtils import FromPreTrained, UNKNOWN

STUDENT = {'name': 'Tal Levy, Lidor Alis',
           'ID': '---, ---'}


def cosine_distance(a, b):
    return np.divide(np.dot(a, b), (np.linalg.norm(a) * np.linalg.norm(b)))


def get_k_nearest(k, anchor, weights, word_to_idx):
    results = []
    embedded_anchor = weights[word_to_idx[anchor]]
    for word, weight in zip(word_to_idx.keys(), weights):
        if not (word == anchor or word == UNKNOWN):
            results.append((cosine_distance(embedded_anchor, weight), word))
    results.sort(key=lambda item: item[0], reverse=True)
    return [(item[1], f'{item[0]:.4f}') for item in results[:k]]


def top_k():
    embed = FromPreTrained('embeddings.txt', 'words.txt')
    weights = embed.get_embeddings()
    word_to_idx = embed.get_word_to_idx()
    for word in ['dog', 'england', 'john', 'explode', 'office']:
        print(get_k_nearest(5, word, weights, word_to_idx))


if __name__ == "__main__":
    top_k()
