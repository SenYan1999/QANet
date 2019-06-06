import torch
import torch.nn as nn
import torch.nn.functional as F
# from config import Config
import config
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle


# config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def f1_score(p1s, p2s, y1s, y2s):
    f1s = []
    for i in range(len(p1s)):
        predicted = 0
        if p1s[i] > y1s[i]:
            predicted = min(y2s[i] - p1s[i], p2s[i] - p1s[i])
            predicted = predicted + 1 if predicted >= 0 else 0
        else:
            predicted = min(p2s[i] - y1s[i], y2s[i] - y1s[i])
            predicted = predicted + 1 if predicted >= 0 else 0
        recall = predicted.item() / (y2s[i].item() - y1s[i].item() + 1)
        precise = predicted.item() / (p1s[i].item() - p2s[i].item() + 1)
        f1s.append(2*recall*precise / (recall + precise))
    return sum(f1s) / len(f1s)


def em(p1s, p2s, y1s, y2s):
    ems = []
    for i in range(len(p2s)):
        ems.append(int(p1s[i] == y1s[i] and p2s[i] == y2s[i]))
    return sum(ems) / len(ems)


def valid(model, data):
    dataloader = DataLoader(data, config.val_batch_size, shuffle=False)
    dataiter = iter(dataloader)
    model.to(device)
    model.eval()
    losses = []
    f1s = []
    ems = []
    num_batch_nums = config.val_batch_size
    with torch.no_grad():
        for batch in num_batch_nums:
            cw, cc, qw, qc, y1s, y2s, ids = next(dataiter)
            cw, cc, qw, qc = cw.to(device), cc.to(device), qw.to(device), qc.to(device)
            loss_1 = F.nll_loss(p1s, y1s)
            loss_2 = F.nll_loss(p2s, y2s)
            loss = (loss_1 + loss_2) / 2
            losses.append(loss.item())
            p1s, p2s = model(cw, cc, qw, qc)
            f1s.append(f1_score(p1s, p2s, y1s, y2s))
            ems.append(em(p1s, p2s, y1s, y2s))
    return np.mean(f1s), np.mean(ems)


def get_input(input, word2idx, char2idx):
    context_tokens, context_chars, question_chars, question_tokens = [], [], [], []
    y1s, y2s = [], []
    ids = []
    for example in tqdm(input):
        context_token = get_word_idx(example['context_tokens'], word2idx, config.para_limit)
        question_token = get_word_idx(example['question_tokens'], word2idx, config.para_limit)
        context_char = get_char_idx(example['context_chars'], char2idx, config.para_limit, config.char_limit)
        question_char = get_char_idx(example['question_chars'], char2idx, config.para_limit, config.char_limit)
        y1 = example['y1s']
        y2 = example['y2s']
        id = example['uuid']
        context_tokens.append(context_token)
        question_tokens.append(question_token)
        context_chars.append(context_char)
        question_chars.append(question_char)
        y1s.append(y1)
        y2s.append(y2)
        ids.append(id)
    return context_tokens, context_chars, question_tokens, question_chars, y1s, y2s, ids


def get_word_idx(input, token2idx, limit):
    length = len(input)
    if(length <= limit):
        for i in range(limit - length):
            input.append('<pad>')
    else:
        input = input[:limit]
    result = [token2idx[x] if x in token2idx else 1 for x in input]
    return result


def get_char_idx(input, token2idx, para_limit, word_limit):
    para_lenght = len(input)
    pad = ['<pad>'] * word_limit
    if para_lenght <= para_limit:
        for i in range(para_limit - para_lenght):
            input.append(pad)
    else:
        input = input[:para_limit]
    for i in range(len(input)):
        if len(input[i]) <= word_limit:
            for _ in range(word_limit - len(input[i])):
                input[i].append('<pad>')
        else:
            input[i] = input[i][:word_limit]
    result = [[token2idx[x] if x in token2idx else 1 for x in word] for word in input]
    return result


class SQuADData(Dataset):
    def __init__(self, data_path, word2idx):
        self.data = pickle.load(open(data_path, 'rb'))[0]

        self.word2idx = word2idx
        self.char2idx = {}
        all_chars = '`1234567890-=qwertyuiop[]\\asdfghjkl;\'zxcvbnm,./~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>?'
        for i, c in enumerate(all_chars):
            self.char2idx[c] = i + 2
        self.char2idx['<pad>'] = 0
        self.char2idx['<unk>'] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        context_token = get_word_idx(data['context_tokens'], self.word2idx, config.para_limit)
        context_char = get_char_idx(data['context_chars'], self.char2idx, config.para_limit, config.char_limit)
        question_token = get_word_idx(data['question_tokens'], self.word2idx, config.ques_limit)
        question_char = get_char_idx(data['question_chars'], self.char2idx, config.ques_limit, config.char_limit)
        y1 = min(data['y1s'], 399)
        y2 = min(data['y2s'], 399)
        id = data['uuid']

        return torch.tensor(context_token, dtype=torch.long, device = device),\
            torch.tensor(context_char, dtype=torch.long, device = device), \
            torch.tensor(question_token, dtype=torch.long, device=device), \
            torch.tensor(question_char, dtype=torch.long, device = device), \
            torch.tensor(y1, dtype=torch.long, device = device), \
            torch.tensor(y2, dtype=torch.long, device = device), \
            torch.tensor(id, dtype=torch.long, device = device)
