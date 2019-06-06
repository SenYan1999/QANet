import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import *

# config = Config()


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super(DepthWiseSeparableConv, self).__init__()

        if dim==1:
            self.depthConv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k,
                                       groups=in_ch, padding= k//2, bias=bias)
            self.pointConv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                                       padding=0, bias=bias)
        elif dim==2:
            self.depthConv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k,
                                       groups=in_ch, padding= k//2, bias=bias)
            self.pointConv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k,
                                       padding=k//2, bias=bias)
        else:
            raise Exception('DepthWiseSeparableConv dim wrong! Expect 1 or 2, but get %d' %dim)

        nn.init.kaiming_normal_(self.depthConv.weight)
        nn.init.constant_(self.depthConv.bias, 0)
        nn.init.kaiming_normal_(self.pointConv.weight)
        nn.init.constant_(self.pointConv.bias, 0)

    def forward(self, input):
        return self.pointConv(self.depthConv(input))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super(Highway, self).__init__()

        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        x = x.transpose(1, 2)

        return x


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.conv2d = DepthWiseSeparableConv(config.d_char, config.d_char, 5, dim=2)
        self.highway = Highway(2, config.d_char+config.d_word)

    def forward(self, char_embed, word_embed):
        char_embed = char_embed.permute(0, 3, 1, 2)
        char_embed = F.dropout(char_embed, config.dropout, training=self.training)
        char_embed = self.conv2d(char_embed)
        char_embed = F.relu(char_embed)
        char_embed = torch.max(char_embed, dim=3)[0]

        word_embed = F.dropout(word_embed, config.dropout)
        word_embed = word_embed.permute(0, 2, 1)

        embed = torch.cat([char_embed, word_embed], dim=1)
        embed = self.highway(embed)

        return embed


class MutiheadAttention(nn.Module):
    def __init__(self):
        super(MutiheadAttention, self).__init__()

        self.d_k = config.d_model // config.n_head
        self.q_weight = nn.Linear(config.d_model, config.d_model)
        self.k_weight = nn.Linear(config.d_model, config.d_model)
        self.v_weight = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.d_model, config.d_model)

    def forward(self, x, mask):
        batch_size, _, x_1 = x.size()
        x = x.transpose(1, 2)
        q = self.q_weight(x).view(batch_size, x_1, config.n_head, self.d_k)
        k = self.k_weight(x).view(batch_size, x_1, config.n_head, self.d_k)
        v = self.v_weight(x).view(batch_size, x_1, config.n_head, self.d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(batch_size*config.n_head, x_1, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(batch_size*config.n_head, x_1, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(batch_size*config.n_head, x_1, self.d_k)
        mask = mask.unsqueeze(1).expand(-1, x_1, -1).repeat(config.n_head, 1, 1)

        atten = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(config.d_model)
        atten = mask_logits(atten, mask)
        atten = F.softmax(atten, dim=2)
        atten = self.dropout(atten)

        out = torch.bmm(atten, v)
        out = out.view(config.n_head, batch_size, x_1, self.d_k).permute(1, 2, 0, 3).contiguous().view(batch_size, x_1, config.n_head*self.d_k)
        out = self.fc(out)
        out = self.dropout(out)

        return out.transpose(1, 2)


class CQAttention(nn.Module):
    def __init__(self):
        super(CQAttention, self).__init__()

        w = torch.empty(config.d_model * 3)
        lim = 1 / config.d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C, Q, c_mask, q_mask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        c_mask = c_mask.unsqueeze(2)
        q_mask = q_mask.unsqueeze(1)

        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)
        S = torch.cat((Ct, Qt, CQ), dim=3)
        S = torch.matmul(S, self.w)
        S1 = F.softmax(mask_logits(S, q_mask), dim=2)
        S2 = F.softmax(mask_logits(S, c_mask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat((C, A, torch.mul(C, A), torch.mul(C, B)), dim=2)
        out = F.dropout(out, p=config.dropout, training=self.training)

        return out.transpose(1, 2)


class PosEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / config.d_model) if i % 2 == 0 else -10000 ** ((1 - i) / config.d_model) for i in range(config.d_model)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(config.d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(config.d_model, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


class EncoderBlock(nn.Module):
    def __init__(self, n_conv, ch_dim, k, length):
        super(EncoderBlock, self).__init__()

        self.conv_layer = nn.ModuleList([DepthWiseSeparableConv(ch_dim, ch_dim, k) for _ in range(n_conv)])
        self.self_atten = MutiheadAttention()
        self.fc = nn.Linear(ch_dim, ch_dim)
        self.norm_layer = nn.LayerNorm([config.d_model, length])
        self.L = length
        self.pos = PosEncoder(length)

    def forward(self, x, mask):
        out = self.pos(x)
        res = out
        out = self.norm_layer(out)

        for i, conv in enumerate(self.conv_layer):
            out = conv(out)
            out = F.relu(out)
            out = res + out

            if i % 2 == 1:
                out = F.dropout(out, config.dropout, training=self.training)

            res = out
            out = self.norm_layer(out)

        out = self.self_atten(out, mask)
        out = res + out
        out = F.dropout(out, config.dropout, training=self.training)
        res = out
        out = self.norm_layer(out)

        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = res + out
        out = F.dropout(out, config.dropout, training=self.training)

        return out


class Pointer(nn.Module):
    def __init__(self):
        super(Pointer, self).__init__()

        w1 = torch.empty(config.d_model*2)
        w2 = torch.empty(config.d_model*2)
        lim = 3 / (2 * config.d_model)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat((M1, M2), dim=1)
        X2 = torch.cat((M1, M3), dim=1)
        Y1 = mask_logits(torch.matmul(self.w1, X1), mask)
        Y2 = mask_logits(torch.matmul(self.w2, X2), mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)

        return p1, p2


class QANet(nn.Module):
    def __init__(self, pretrained_em, config):
        super(QANet, self).__init__()

        self.char_embedding = nn.Embedding(96, config.d_char, padding_idx=0)
        self.word_embedding = nn.Embedding.from_pretrained(pretrained_em)
        self.word_embedding.weight.requires_grad=False
        self.word_embedding.weight[1].requires_grad=True
        self.embed = Embedding()
        self.context_conv = DepthWiseSeparableConv(config.d_word+config.d_char, config.d_model, 5)
        self.question_conv = DepthWiseSeparableConv(config.d_word+config.d_char, config.d_model, 5)
        self.c_emb_enc = EncoderBlock(4, config.d_model, 7, config.para_limit)
        self.q_emb_enc = EncoderBlock(4, config.d_model, 7, config.ques_limit)
        self.cqatten = CQAttention()
        self.resizer = DepthWiseSeparableConv(config.d_model * 4, config.d_model, 5)
        enc_blk = EncoderBlock(n_conv=2, ch_dim=config.d_model, k=5, length=config.para_limit)
        self.enc_blks = nn.ModuleList([enc_blk] * 7)
        self.out = Pointer()

    def forward(self, context_word, context_char, question_word, question_char):
        cw_mask = (torch.zeros_like(context_word) != context_word).float()
        qw_mask = (torch.zeros_like(question_word) != question_word).float()

        cw_embed, cc_embed = self.word_embedding(context_word), self.char_embedding(context_char)
        qw_embed, qc_embed = self.word_embedding(question_word), self.char_embedding(question_char)
        C, Q = self.embed(cc_embed, cw_embed), self.embed(qc_embed, qw_embed)
        C, Q = self.context_conv(C), self.question_conv(Q)
        Ce = self.c_emb_enc(C, cw_mask)
        Qe = self.q_emb_enc(Q, qw_mask)

        X = self.cqatten(Ce, Qe, cw_mask, qw_mask)
        M1 = self.resizer(X)
        for enc in self.enc_blks:
            M1 = enc(M1, cw_mask)
        M2 = M1
        for enc in self.enc_blks:
            M2 = enc(M2, cw_mask)
        M3 = M2
        for enc in self.enc_blks:
            M3 = enc(M3, cw_mask)
        p1, p2 = self.out(M1, M2, M3, cw_mask)

        return p1, p2