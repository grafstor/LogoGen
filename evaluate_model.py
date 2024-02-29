
import math
import time
import random
import json
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from dataset import Dataset
import converter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EMB_LENGTH = 11
DOTS_LENGTH = 11
MAX_STRING_LENGTH = 10
MAX_SVG_LENGTH = 250


class EncoderRNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.embedding = nn.Embedding(input_size, EMB_LENGTH)
        self.gru = nn.GRU(EMB_LENGTH, EMB_LENGTH)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, EMB_LENGTH)

class AttnDecoderRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.Linear(DOTS_LENGTH * 2, MAX_SVG_LENGTH)
        self.attn_combine = nn.Linear(EMB_LENGTH + DOTS_LENGTH, DOTS_LENGTH)
        self.gru = nn.GRU(DOTS_LENGTH, DOTS_LENGTH)
        self.out = nn.Linear(DOTS_LENGTH, DOTS_LENGTH)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        hidden = hidden[0]

        cat = torch.cat((input, hidden), 1)

        attn_weights = torch.tanh(self.attn(cat))

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((input, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)


        output = F.relu(output)

        hidden = hidden.unsqueeze(0)

        output, hidden = self.gru(output, hidden)

        output = torch.tanh(self.out(output[0]))

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, DOTS_LENGTH)


def load_model(encoder, decoder, encoder_optimizer, decoder_optimizer, filename):
    checkpoint = torch.load(filename)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    print("Model loaded successfully!")

def evaluate(encoder, decoder, sentence, i):
    w = sentence[0]
    sentence = [word2index[w] for w in sentence]
    sentence = torch.tensor(sentence, dtype=torch.long).view(-1, 1)

    with torch.no_grad():
        input_tensor = torch.tensor(sentence, dtype=torch.long).view(-1, 1)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(MAX_SVG_LENGTH, EMB_LENGTH)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor(SOS_token)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(MAX_SVG_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            decoded_words.append(decoder_output[0].detach().numpy())

            decoder_input = decoder_output[0]

        try:
            svg = c.to_svg(decoded_words)
            c.save(f'evaluate/{i}-{w}.svg', svg)
        except Exception as e:
            print('CANNOT SAVE')
            print(e)

        return decoded_words


def load_word2index_from_json(filename):
    with open(filename, 'r') as f:
        word2index = json.load(f)
    print("Word2index dictionary loaded successfully from JSON!")
    return word2index

c = converter.Converter()

word2index = load_word2index_from_json('word2index.json')
uniq_words = len(word2index)

learning_rate = 0.01

SOS_token = [0,0,0,0,0,0,0,0,0,0,-1]
EOS_token = [0,0,0,0,0,0,0,0,0,0,1]

teacher_forcing_ratio = 0.5

encoder = EncoderRNN(uniq_words)
decoder = AttnDecoderRNN()
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

load_model(encoder, decoder, encoder_optimizer, decoder_optimizer, 'model/model_weights_200.pth')


evaluate(encoder, decoder, ['blade'], 1)

