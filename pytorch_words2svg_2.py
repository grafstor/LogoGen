

'''

forward ==============================
forward input torch.Size([1, 1])
forward hidden torch.Size([1, 1, 256])
forward encoder_outputs torch.Size([10, 256])

forward embedded torch.Size([1, 1, 256])
forward cat torch.Size([1, 512])
forward attn torch.Size([1, 10])
forward attn_weights torch.Size([1, 10])

forward aw_unsqueeze torch.Size([1, 1, 10])
forward eo_unsqueeze torch.Size([1, 10, 256])
forward attn_applied torch.Size([1, 1, 256])
forward output cat torch.Size([1, 512])
forward output attn_combine torch.Size([1, 256])
forward output unsqueeze torch.Size([1, 1, 256])
forward output gru torch.Size([1, 1, 256])
forward hidden gru torch.Size([1, 1, 256])
forward output log_softmax torch.Size([1, 1699])




input_tensor torch.Size([8, 1])
target_tensor torch.Size([5, 1])
max_length 10
encoder_hidden torch.Size([1, 1, 256])
input_length 8
target_length 5
encoder_outputs torch.Size([10, 256])
decoder_input torch.Size([1, 1])
===================================
input_tensor torch.Size([8, 1])
target_tensor torch.Size([8, 1])
max_length 10
encoder_hidden torch.Size([1, 1, 256])
input_length 8
target_length 8
encoder_outputs torch.Size([10, 256])
decoder_input torch.Size([1, 1])

'''

import math
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from dataset import Dataset
import converter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.switch_backend('agg')

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


SOS_token = [0,0,0,0,0,0,0,0,0,0,-1]
EOS_token = [0,0,0,0,0,0,0,0,0,0,1]

teacher_forcing_ratio = 0.5

def mse(y_true, y_pred):
    error = np.mean(np.power(y_true - y_pred, 2))
    return error

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def evaluate(encoder, decoder, sentence, i):
    w = sentence[0]
    sentence = [dataset.word2index[w] for w in sentence]
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
            c.save(f'result/{i}-{w}.svg', svg)
        except:
            pass

        return decoded_words

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_SVG_LENGTH, EMB_LENGTH)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor(SOS_token)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output[0],target_tensor[di])
            decoder_input = target_tensor[di] 

    else:

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

            decoder_input = decoder_output[0]

            loss += criterion(decoder_output[0],target_tensor[di])

            if decoder_input.detach().numpy()[-1]>0.3:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def tensorsFromPair(pair):
    return (torch.tensor(pair[0], dtype=torch.long).view(-1, 1), torch.tensor(pair[1]))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainIters(encoder, decoder, n_iters, print_every=100, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []

    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

    # criterion = nn.NLLLoss()
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            evaluate(encoder, decoder, ['shirt'], iter)
            evaluate(encoder, decoder, ['box'], iter)

            save_model(encoder, decoder, encoder_optimizer, decoder_optimizer, f'model/model_weights_{iter}.pth')

    showPlot(plot_losses)

def save_model(encoder, decoder, encoder_optimizer, decoder_optimizer, filename):
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print("Model saved successfully!")

def save_word2index_to_json(word2index, filename):
    with open(filename, 'w') as f:
        json.dump(word2index, f)
    print("Word2index dictionary saved successfully to JSON!")

with open('folders.txt', 'r') as f:
    folders = f.read().split('\n')
    folders = [f for f in folders if f]
    folders = folders[:500]

# folders = [
#     'data/testdataset'
# ]

c = converter.Converter()

dataset = Dataset(MAX_SVG_LENGTH, folders)

save_word2index_to_json(dataset.word2index, 'word2index.json')

pairs = dataset.get_pairs()
print(len(pairs[0][1][0]))

encoder1 = EncoderRNN(len(dataset.uniq_words))
attn_decoder1 = AttnDecoderRNN()

trainIters(encoder1, attn_decoder1, 10000)


