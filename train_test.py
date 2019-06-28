import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import ujson as json
import config
from standardmodel import QANet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from math import log2

class SQuADData(Dataset):
    def __init__(self, npz_file):
        super().__init__()
        data = np.load(npz_file)
        self.context_idxs = torch.from_numpy(data["context_idxs"]).long().to(device)
        self.context_char_idxs = torch.from_numpy(data["context_char_idxs"]).long().to(device)
        self.ques_idxs = torch.from_numpy(data["ques_idxs"]).long().to(device)
        self.ques_char_idxs = torch.from_numpy(data["ques_char_idxs"]).long().to(device)
        self.y1s = torch.from_numpy(data["y1s"]).long().to(device)
        self.y2s = torch.from_numpy(data["y2s"]).long().to(device)
        self.ids = torch.from_numpy(data["ids"]).long().to(device)

    def __len__(self):
        return self.context_idxs.shape[0]

    def __getitem__(self, item):
        return self.context_idxs[item], self.context_char_idxs[item], self.ques_idxs[item], self.ques_char_idxs[item], self.y1s[item], self.y2s[item], self.ids[item]


# prepare data
print('prepare data')
# config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('pre_data/word_emb.json') as fh:
    word_mat = np.array(json.load(fh), dtype=np.float32)
print('loading train_dataset')
train_dataset = SQuADData('pre_data/train.npz')

# define model
print('define model')
model = QANet(word_mat)
model = model.to(device)
lr = config.learning_rate
base_lr = 1.0
warm_up = config.lr_warm_up_num
cr = lr / log2(warm_up)
optimizer = torch.optim.Adam(lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps,
                             weight_decay=3e-7, params=model.parameters())
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: cr * log2(ee + 1) if ee < warm_up else lr)

print('begin train')
for epoch in range(config.num_epoch):
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = iter(train_dataloader)
    losses = []
    model.train()
    for step in tqdm(range(len(train_dataset) // config.batch_size)):
        optimizer.zero_grad()
        cw, cc, qw, qc, y1s, y2s, ids = next(train_iter)
        p1, p2 = model(cw, cc, qw, qc)
        loss_1 = F.nll_loss(p1, y1s, reduction='mean')
        loss_2 = F.nll_loss(p2, y2s, reduction='mean')
        loss = (loss_1 + loss_2) / 2
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
