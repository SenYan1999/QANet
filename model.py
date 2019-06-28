import torch
import torch.nn as nn
import torch.nn.functional as F
import math

d_model = 128
d_char_embed = 200
d_word_embed = 300
d_embed = d_char_embed + d_word_embed
num_chars = 96
dropout_p = 0.5
num_head = 8
para_limit = 400
ques_limit = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QANet(nn.Module):
    def __init__(self, word_pretrained: torch.Tensor):
        super(QANet, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(torch.Tensor(word_pretrained), padding_idx=0)
        # self.word_embed.weight.requires_grad = False
        # self.word_embed.weight[1].required_grad = True
        self.char_embed = nn.Embedding(num_chars, d_char_embed, padding_idx=0)
        self.embed_highway = HighwayNetwork(2, d_embed)
        self.context_embed_enc = EncoderBlock(4, d_embed, d_model, 7, num_head, para_limit)
        self.question_embed_enc = EncoderBlock(4, d_embed, d_model, 7, num_head, ques_limit)
        self.cqAttention = CQAttention()
        self.resizer = DepthwiseSeparableCNN(4 * d_model, d_model, 5)
        self.dec = nn.ModuleList([EncoderBlock(2, d_model, d_model, 5, num_head, para_limit) for _ in range(7)])
        self.out = Pointer()

    def forward(self, cw, cc, qw, qc):
        cw_mask = (torch.zeros_like(cw) != cw).to(torch.float)
        qw_mask = (torch.zeros_like(qw) != qw).to(torch.float)

        cw, cc = self.word_embed(cw), self.char_embed(cc)
        qw, qc = self.word_embed(qw), self.char_embed(qc)

        cc, qc = torch.max(cc, dim=-2)[0], torch.max(qc, dim=-2)[0]
        C, Q = torch.cat((cw, cc), dim=-1), torch.cat((qw, qc), dim=-1)
        C, Q = self.embed_highway(C), self.embed_highway(Q)

        # input layer: B * S -> B * S * d_embed
        # C, Q = self.embed(cw, cc), self.embed(qw, qc)
        # embedding layer: B * S * d_embed -> B * S * d_model
        C, Q = self.context_embed_enc(C, cw_mask), self.question_embed_enc(Q, qw_mask)
        # cq attention: B * S * d_model -> B * S * (4*d_model)
        cq = self.cqAttention(C, Q, cw_mask, qw_mask)
        # get M1, M2, M3 decode
        cq = self.resizer(cq)
        M1 = cq
        for dec in self.dec:
            M1 = dec(M1, cw_mask)
        M2 = M1
        for dec in self.dec:
            M2 = dec(M2, cw_mask)
        M3 = M2
        for dec in self.dec:
            M3 = dec(M3, cw_mask)
        p1, p2 = self.out(M1, M2, M3, cw_mask)
        return p1, p2


class HighwayNetwork(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super(HighwayNetwork, self).__init__()
        self.layer_num = layer_num
        self.T = nn.Linear(size, size, bias=True)
        self.H = nn.Linear(size, size, bias=True)

    def forward(self, x):
        for i in range(self.layer_num):
            h, t = F.relu(self.H(x)), torch.sigmoid(self.T(x))
            out = h * t + x * (1 - t)
        return F.dropout(out, p=dropout_p, training=self.training)


class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, in_c, out_c, k, dimension=1, bias=True):
        super(DepthwiseSeparableCNN, self).__init__()
        if dimension == 1:
            self.depthwise = nn.Conv1d(in_channels=in_c, out_channels=in_c, groups=in_c,
                                       kernel_size=k, padding=(k - 1) // 2, bias=bias)
            self.separable = nn.Conv1d(in_channels=in_c, out_channels=out_c,
                                       kernel_size=1, padding=0, bias=bias)
        # initialize the parameters
        # nn.init.kaiming_normal_(self.depthwise.weight)
        # nn.init.constant_(self.depthwise.bias, 0.01)
        # nn.init.kaiming_normal_(self.separable.weight)
        # nn.init.constant_(self.separable.bias, 0.01)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.separable(self.depthwise(x)).permute(0, 2, 1)


## WARNING MASK !!!!
## PAD OUTPUT NOT 0
class MutiheadSelfAttention(nn.Module):
    def __init__(self, num_head, dim):
        super(MutiheadSelfAttention, self).__init__()
        self.num_head = num_head
        self.q_linear = nn.ModuleList([nn.Linear(dim, dim // num_head) for _ in range(num_head)])
        self.k_linear = nn.ModuleList([nn.Linear(dim, dim // num_head) for _ in range(num_head)])
        self.v_linear = nn.ModuleList([nn.Linear(dim, dim // num_head) for _ in range(num_head)])
        self.fc = nn.Linear(dim, dim)
        self.a = 1 / math.sqrt(dim // 8)

    def forward(self, x, x_mask):
        # x: B * S * d_model
        attentions = []
        mask = x_mask.unsqueeze(1).expand(-1, x.shape[1], -1).to(torch.int32)
        mask = (mask & mask.transpose(1, 2)).to(torch.float)
        for i in range(self.num_head):
            q = self.q_linear[i](x)
            k = self.k_linear[i](x)
            v = self.v_linear[i](x)
            atten = torch.bmm(q, k.transpose(1, 2)) * self.a
            atten = mask_logits(atten, mask)
            atten = F.softmax(atten, dim=-1)
            attentions.append(torch.bmm(atten, v))  # B * S * d_model
        out = torch.cat(attentions, dim=-1)
        out = self.fc(out)
        F.dropout(out, p=dropout_p, training=self.training)
        return out


class PosEncoder(nn.Module):
    def __init__(self, length, d_in):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / d_in) if i % 2 == 0 else -10000 ** ((1 - i) / d_in) for i in
             range(d_in)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_in)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_in, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)),
                                         requires_grad=False).permute(1, 0).to(device)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_conv, in_ch, out_ch, k, num_head, length):
        super(EncoderBlock, self).__init__()
        self.pos = PosEncoder(length, in_ch)
        self.convLayers = nn.ModuleList([DepthwiseSeparableCNN(in_ch, out_ch, k) if _ == 0
                               else DepthwiseSeparableCNN(out_ch, out_ch, k) for _ in range(num_conv - 1)])
        self.selfAttention = MutiheadSelfAttention(num_head, out_ch)
        self.out = nn.Linear(out_ch, out_ch)
        self.normcs = nn.ModuleList([nn.LayerNorm([length, out_ch]) for _ in range(num_conv - 1)])
        self.normAtten = nn.LayerNorm([length, out_ch])
        self.normOut = nn.LayerNorm([length, out_ch])

    def forward(self, x, x_mask):
        x = self.pos(x)
        # x: B * S * (p1 + p2)
        for i, conv in enumerate(self.convLayers):
            res = x
            x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=dropout_p, training=self.training)
            if i > 0:
                x = x + self.normcs[i - 1](res)
        # x: B * S * d_model
        res = x
        x = self.selfAttention(x, x_mask)
        x = F.dropout(x, p=dropout_p, training=self.training)
        x = x + self.normAtten(res)
        res = x
        x = self.out(x)
        x = F.dropout(x, p=dropout_p, training=self.training)
        x = x + self.normOut(res)
        return x


class CQAttention(nn.Module):
    def __init__(self):
        super(CQAttention, self).__init__()
        self.out = nn.Linear(d_model * 3, 1)

    def forward(self, C, Q, c_mask, q_mask):
        mask_c = c_mask.unsqueeze(2).expand(-1, -1, Q.shape[1]).to(torch.int32)
        mask_q = q_mask.unsqueeze(1).expand(-1, C.shape[1], -1).to(torch.int32)
        mask = (mask_c & mask_q).to(torch.float)
        C_, Q_ = C.unsqueeze(2), Q.unsqueeze(1)
        shape = (C.shape[0], C.shape[1], Q.shape[1], C.shape[2])
        C_, Q_ = C_.expand(shape), Q_.expand(shape)
        S = self.out(torch.cat((Q_, C_, torch.mul(C_, Q_)), dim=-1)).squeeze()
        S = mask_logits(S, mask)
        S1 = torch.softmax(S, dim=1)
        S2 = torch.softmax(S, dim=2)
        C2Q = torch.bmm(S1, Q)
        Q2C = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, C2Q, torch.mul(C, C2Q), torch.mul(C, Q2C)], dim=2)  # B * 400 * (4 * d_model)
        out = F.dropout(out, p=dropout_p, training=self.training)
        return out


class Pointer(nn.Module):
    def __init__(self):
        super(Pointer, self).__init__()
        self.out_1 = nn.Linear(2 * d_model, 1)
        self.out_2 = nn.Linear(2 * d_model, 1) 
        # self.out_1.weight.data.fill_(0.05)
        # self.out_2.weight.data.fill_(0.05)

    def forward(self, M1, M2, M3, mask):
        p1 = self.out_1(torch.cat([M1, M2], dim=-1)).squeeze()
        p2 = self.out_2(torch.cat([M1, M3], dim=-1)).squeeze()
        p1 = mask_logits(p1, mask)
        p2 = mask_logits(p2, mask)
        p1 = F.log_softmax(p1, dim=-1)
        p2 = F.log_softmax(p2, dim=-1)
        return p1, p2

def mask_logits(x, x_mask):
    return x * x_mask + (-1e30) * (1 - x_mask)
