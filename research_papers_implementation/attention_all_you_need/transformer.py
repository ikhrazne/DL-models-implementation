import math

import numpy
import torch
import torch.nn as nn
import torch.functional as f
import numpy as np
from math import sin, cos
from numpy import linalg, matrix


class PositionalEncoding:

    def encode(self, X, d_model):
        result = []

        for position, token in enumerate(X):
            encoded_token = []

            for i, coordinate in enumerate(token):

                if i % 2 == 0:
                    encoded_token.append(sin(position / (10000 ^ (2*i / d_model))))
                else:
                    encoded_token.append(sin(position / (10000 ^ (2*i / d_model))))

            result.append(encoded_token)

        return torch.tensor(result)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dimension: int, vector_size: int):
        super(ScaledDotProductAttention, self).__init__()
        self.query_linear = nn.Linear(vector_size, vector_size)
        self.key_linear = nn.Linear(vector_size, vector_size)
        self.value_linear = nn.Linear(vector_size, vector_size)
        self.dimension = dimension

    def forward(self, X):
        queries = []
        keys = []
        values = []

        for vector in X:
            queries.append(self.query_linear(vector))
            keys.append(self.key_linear(vector))
            values.append(self.value_linear(vector))

        queries = numpy.array(queries)
        keys = numpy.array(keys)
        values = numpy.array(values)

        output_for_softmax = matrix.dot(queries, matrix.transpose(keys)) / math.sqrt(self.dimension)
        attention = matrix.dot(numpy.array(torch.softmax(torch.tensor(output_for_softmax), dim=1)), values)

        return attention


class MultiHeadAttention(nn.Module):

    def __init__(self, vector_size:int, head_number=8):
        super(MultiHeadAttention, self).__init__()
        self.attentions = []

        for i in range(head_number):
            self.attentions.append(ScaledDotProductAttention(vector_size))

    def forward(self, X):
        multi_head_attention = None

        for index, attention in enumerate(self.attentions):

            if multi_head_attention == None:
                multi_head_attention = attention(X)
            else:
                multi_head_attention += attention(X)

        return multi_head_attention


class FeedForwardNN(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(FeedForwardNN, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size * 3)
        self.linear2 = nn.Linear(input_size, input_size)

    def forward(self, X):

        X = torch.relu(self.linear1(X))

        return self.linear2(X)


class Encoder(nn.Module):

    def __init__(
            self,
            vocabulary_size: int,
            model_size: int,
            head: int
    ):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.input_embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                            embedding_dim=model_size)
        self.position_embedding = PositionalEncoding()
        self.multi_head_attention = MultiHeadAttention(model_size, model_size)
        self.feed_forward_net = FeedForwardNN(model_size, model_size)

    def forward(self, X):
        X = self.input_embedding(X)
        encoded_X = self.position_embedding.encode(X, self.model_size)
        X = self.multi_head_attention(encoded_X)
        X = torch.layer_norm(X + encoded_X, normalized_shape=X.size())
        output = []
        for vector in X:
            output.append(self.feed_forward_net(vector))

        return torch.tensor(output)


class Decoder(nn.Module):

    def __init__(self, vocabulary_size, model_size, head):
        super(Decoder, self).__init__()
        self.encoder_component = Encoder(vocabulary_size=vocabulary_size, model_size=model_size, head=head)
        self.linear = nn.Linear(model_size, model_size)

    def forward(self, X, y):
        pass
