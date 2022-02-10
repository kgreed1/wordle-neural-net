from tkinter import Y
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from wordle import Vocab, WordleGame, read_vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Netty(nn.Module):
    def __init__(self, n_inputs, vocab_size):
        super(Netty, self).__init__()
        self.hidden1 = nn.Linear(n_inputs, vocab_size)
        # self.hidden1 = nn.Linear(n_inputs, n_inputs)
        # self.hidden2 = nn.Linear(n_inputs, n_inputs)
        # self.hidden3 = nn.Linear(n_inputs, vocab_size)
        self.activation = nn.Softmax()

    def forward(self, inputs):
        X = self.hidden1(inputs)
        # X = self.hidden2(X)
        # X = self.hidden3(X)
        X = self.activation(X)
        return X

def train_model(model, vocab, vocab_list):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1):
        for i, word in enumerate(tqdm(vocab_list)):
            game = WordleGame(vocab, word)
            y = torch.Tensor([i]).to(torch.long).to(device)

            playing = True
            while playing:
                inputs = game.game_state
                inputs = torch.flatten(inputs).to(device)
                optimizer.zero_grad()
                y_hat = model(inputs)
                y_hat = y_hat.view(1,-1)
                loss = criterion(input=y_hat, target=y)
                loss.backward()
                optimizer.step()
                guess = vocab.index2word[int(torch.argmax(y_hat))]
                playing = game.guess_word(guess)


def play_with_model(model, game):
    playing = True
    while playing:
        inputs = game.game_state
        inputs = torch.flatten(inputs)
        y_hat = model(inputs)
        y_hat = y_hat.view(1,-1)
        guess = vocab.index2word[int(torch.argmax(y_hat))]
        playing = game.guess_word(guess)
        game.print_game_state()
        print('')    

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
    model = Netty(6*5*27*3, len(vocab_list))
    model.load_state_dict(torch.load(path))
    return model

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = read_vocab('vocab.txt', 'vocab')
    vocab_list = list(vocab.word2index.keys())

    model = Netty(6*5*27*3, len(vocab_list)).to(device)
    train_model(model, vocab, vocab_list)
    save_model(model, 'model')

    # model = load_model('model')
    # play_with_model(model, WordleGame(vocab, 'aahed'))
