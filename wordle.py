import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def read_vocab(filepath, name):
    words = open(filepath).read().strip().split('\n')
    vocab = Vocab(name)
    for w in words:
        vocab.add_word(w)
    return vocab

def l_to_n(letter):
    return ord(letter) - 96

def n_to_l(num):
    return chr(num+96)

class WordleGame:
    def __init__(self, vocab, target_word):
        self.vocab = vocab
        self.target_word = target_word
        self.target_index = self.vocab.word2index[target_word]
        # 6 stages, 5 letters, 26 letters + null, B/Y/G
        self.game_state = torch.zeros(6,5,27,3, device=device)
        self.turn = 0
        self.won = 0


    def guess_word(self, guessed_word):

        if guessed_word not in self.vocab.word2index.keys():
            print('Incorrect word')
            return False

        if guessed_word == self.target_word:
            self.won = True

        guess_ls = list(guessed_word.lower())
        target_ls = list(self.target_word.lower())
        guess_l_idxes = [l_to_n(l) for l in guess_ls]
        target_l_idxes = [l_to_n(l) for l in target_ls]
        target_counts = Counter(target_l_idxes)

        guess_feedback = torch.zeros(5, 27, 3)

        # first check correct letter correct place
        for i,(g,t) in enumerate(zip(guess_l_idxes, target_l_idxes)):
            # if correct letter correct place
            if g == t:
                guess_feedback[i][g][2] = 1
                if target_counts[g] == 1:
                    del target_counts[g]
                else:
                    target_counts[g] -= 1

        # check correct letter incorrect place
        for i,(g,t) in enumerate(zip(guess_l_idxes, target_l_idxes)):
            if g in target_counts.keys() and guess_feedback[i][g][2] != 1:
                guess_feedback[i][g][1] = 1

                if target_counts[g] == 1:
                    del target_counts[g]
                else:
                    target_counts[g] -= 1

        # incorrect letter
        for i,(g,t) in enumerate(zip(guess_l_idxes, target_l_idxes)):
            if guess_feedback[i][g][2] != 1 and guess_feedback[i][g][1] != 1:
                guess_feedback[i][g][0] = 1
            
        self.game_state[self.turn] = guess_feedback
        self.turn += 1
        
        return not self.won and self.turn < 6

    def print_game_state(self):
        string = ''
        feedback_dict = {2: 'G', 1: 'Y', 0: 'B'}
        for guess in self.game_state:
            l_str = ''
            f_str = ''
            for letter in guess:
                l_idx = int(torch.argmax(letter)) // 3
                l = n_to_l(l_idx)
                f_idx = int(torch.argmax(letter) - l_idx * 3)
                feedback = feedback_dict[f_idx]
                l_str += l
                f_str += feedback
            string += l_str + ' ' + f_str + '\n'
        print(string)

if __name__=="__main__":
    vocab = read_vocab('vocab.txt', 'vocab')
    game = WordleGame(vocab, 'humor')
    playing = True
    while playing:
        guess = input('Next guess: ')
        print('')
        playing = game.guess_word(guess)
        game.print_game_state()
        print('')