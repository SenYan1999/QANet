import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
from model import QANet
# from config import Config
import config
from standardmodel import QANet
from utils import SQuADData
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import valid
import torch.optim as optim
from math import log2


# prepare data
print('prepare data')
# config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pre_trained_ = pickle.load(open('pre_data/embed_pre.pkl', 'rb'))
word2idx = pre_trained_[1]
pre_trained = torch.tensor(pre_trained_[0])
del pre_trained_
print('loading train_dataset')
train_dataset = SQuADData('pre_data/data_train_pre.pkl', word2idx)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
train_iter = iter(train_dataloader)
# dev_dataset = SQuADData('pre_data/dev_pre.pkl')

# define model
print('define model')
model = QANet(torch.tensor(pre_trained))
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
    losses = []
    for step in tqdm(range(len(train_dataset) // config.batch_size)):
        optimizer.zero_grad()
        cw, cc, qw, qc, y1s, y2s, ids = next(train_iter)
        p1, p2 = model(cw, cc, qw, qc)
        loss_1 = F.nll_loss(p1, y1s, reduction='mean')
        loss_2 = F.nll_loss(p2, y2s, reduction='mean')
        loss = (loss_1 + loss_2) / 2
        loss.backward()
        optimizer.step()
        if(step % 100 == 0):
            print('Epoch: %2d | Step: %3d | Loss: %3f' % (epoch, step, loss))
    # f1, em = valid(dev_dataset)
    # print('-' * 30)
    # print('Valid:')
    # print('F1: %.2f | EM: %.2f | LOSS: %.2f' % (f1, em, np.mean(losses)))
