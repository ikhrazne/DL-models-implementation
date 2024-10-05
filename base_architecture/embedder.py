import torch
import torch.nn as nn
import torch.optim as optim


class WordEmbedder(nn.Module):

    def __init__(self):
        super(WordEmbedder, self).__init__()


if __name__ == "__main__":

    sentences = [
        "I love cats",
        "I hate dogs",
        "Cats are cute",
        "Dogs are loyal"
    ]

    vocab = list(set(word for sentence in sentences for word in sentence.split()))

    word_to_index = {word: i for i, word in enumerate(vocab)}

    sentences_indices = [
        [word_to_index[word] for word in sentence.split()]
        for sentence in sentences
    ]

    embedder = nn.Embedding(len(vocab), 10)

    sentence = "I love mohamed"

    sentences_index = [word_to_index[word] for word in sentence.split()]

    embedded = embedder(torch.tensor(sentences_index))

    print(embedded)


