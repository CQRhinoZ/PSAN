import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# seed = 1
# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(1)
class FeedForwardNet(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x, residual):
        out = self.dropout1(x)
        out = self.ln1(residual + out)

        residual = out
        out = self.ffn(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out
class GatedMechanism(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.gate1 = nn.Conv2d(in_channels=emb_dim*2, out_channels=emb_dim, kernel_size=(1,1), stride=1, groups=emb_dim)
        self.gate2 = nn.Conv2d(in_channels=emb_dim*2, out_channels=emb_dim, kernel_size=(1,1), stride=1, groups=emb_dim)

    def forward(self, x_out, x):
        xx = torch.cat([x, x_out], dim=1)
        sigmoid_gate = self.gate1(xx)
        tanh_gate = self.gate2(xx)
        sigmoid_gate = torch.sigmoid(sigmoid_gate)
        tanh_gate = torch.sigmoid(tanh_gate)
        out = sigmoid_gate * x_out + (1 - sigmoid_gate) * tanh_gate
        return out

class DataEmbedding(nn.Module):
    def __init__(self, emb_dim, in_len, out_len, nnodes,
                 use_tod=True, use_dow=True, use_adp=True,
                 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.in_len = in_len
        self.out_len = out_len
        self.nnodes = nnodes
        self.use_tod = use_tod
        self.use_dow = use_dow
        self.use_adp = use_adp

        self.input_embedding = nn.Linear(3, emb_dim)

        # self.TOD = nn.init.xavier_uniform_(nn.Parameter(torch.empty(288, emb_dim)))
        # self.DOW = nn.init.xavier_uniform_(nn.Parameter(torch.empty(7, emb_dim)))

        self.SE = nn.init.xavier_uniform_(nn.Parameter(torch.empty(in_len, nnodes, emb_dim)))

        self.input_proj = nn.Linear(2*self.emb_dim, self.emb_dim)



    def forward(self, x):
        """
        :param x: BxTxNxF(flow, day, week)
        :return: BxTxNxD
        """
        x1 = self.input_embedding(x)
        x2 = self.SE.expand(x.shape[0], *self.SE.shape)
        # x4 = tp_enc_2d(x1)
        x = torch.cat([x1, x2], dim=-1) # train
        # 新的加入了STPE

        return self.input_proj(F.dropout(x, p=0.1))

class SpatialMHSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.model_dim = embed_dim
        self.num_heads = num_heads
        self.mhssa = MHSSA(embed_dim, num_heads)

    def forward(self, x, sim=None):
        """
        :param x: [B, T, N, D]
        :return: [B, T, N, D]
        """
        x = self.mhssa(x, sim)
        return x

class GlobalTCN(nn.Module):
    def __init__(self, window_size, emb_dim):
        super().__init__()
        self.window_size = window_size
        self.emb_dim = emb_dim

        self.cc = nn.Conv2d(
            in_channels=emb_dim,
            out_channels=emb_dim,
            kernel_size=(1,window_size),
            stride=1, groups=emb_dim
        )
        self.gated_fusion = GatedMechanism(emb_dim)

        self.fc = nn.Conv2d(
            in_channels=emb_dim,
            out_channels=emb_dim,
            kernel_size=(1, 1),
            stride=1
        )

    def forward(self, x):
        """
        :param x: BxDxNxT
        :return: BxDxNxT
        """
        x_raw = x
        x = F.pad(x, (0, self.window_size, 0, 0)) # BxDxNx(T+T)
        out = self.cc(x[:,:,:,:-1])
        out = self.gated_fusion(out, x_raw)
        return self.fc(out)

class LocalTCN(nn.Module):
    def __init__(self, window_size, emb_dim, period):
        super().__init__()
        self.window_size = window_size
        self.emb_dim = emb_dim
        self.period = period

        self.cc = nn.Conv2d(
            in_channels=emb_dim,
            out_channels=emb_dim,
            kernel_size=(1, period),
            stride=1, groups=emb_dim
        )

        self.fc = nn.Conv2d(
            in_channels=emb_dim,
            out_channels=emb_dim,
            kernel_size=(1, 1),
            stride=1
        )

    def forward(self, x):
        """
        :param x: BxDxNxT
        :return: BxDxNxT
        """
        x = F.pad(x, (0, self.period-1, 0, 0))
        out = self.cc(x)
        out = self.fc(out)
        return out

class MHSSA(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = self.model_dim // num_heads

        self.FC_Q = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.model_dim, self.model_dim)))
        self.FC_K = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.model_dim, self.model_dim)))
        self.FC_V = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.model_dim, self.model_dim)))
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.model_dim, self.model_dim)))

    def forward(self, x, sim=None):
        B = x.shape[0]

        query = x @ self.FC_Q
        key = x @ self.FC_K
        value = x @ self.FC_V
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        att_score = (
                query @ key.transpose(-1, -2)
        ) / self.head_dim ** 0.5


        if sim is not None:
            att_score = att_score + sim.unsqueeze(0).unsqueeze(0)
        # print(key.shape)

        att_score = self.dropout(torch.softmax(att_score, dim=-1))
        out = att_score @ value

        out = torch.cat(
            torch.split(out, B, dim=0), dim=-1
        )
        out = out @ self.out_proj

        return out

class BiLSTCM(nn.Module):
    def __init__(self, window_size, emb_dim):
        super().__init__()
        self.window_size = window_size
        self.emb_dim = emb_dim
        self.r_gtcn = GlobalTCN(window_size, emb_dim)
        self.l_gtcn = GlobalTCN(window_size, emb_dim)

        self.r_ltcn = LocalTCN(window_size, emb_dim, 3)
        self.l_ltcn = LocalTCN(window_size, emb_dim, 3)

        self.fc = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, x):
        """
        :param x: BxTxNxD
        :return: BxTxNxD
        """
        x = x.transpose(1, 3)
        re_x = x.flip(-1)

        # Long Term Feature Extraction Branch
        forward_out = self.r_gtcn(x)
        backward_out = self.l_gtcn(re_x).flip(-1)
        g_out = forward_out + backward_out

        # Short Term Feature Extraction Branch
        forward_out = self.r_ltcn(x)
        backward_out = self.l_ltcn(re_x).flip(-1)
        l_out =  forward_out + backward_out

        out = torch.cat([g_out, l_out], dim=1).transpose(1, 3) # BxTxNx(2xD)
        # out = torch.cat([g_out], dim=1).transpose(1, 3) # BxTxNx(2xD)
        out = self.fc(out)
        return out

class PLM(nn.Module):
    def __init__(self, emb_dim, nnodes, k):
        super().__init__()
        self.nnodes = nnodes
        self.k = k
        self.proto = nn.init.xavier_uniform_(nn.Parameter(torch.empty(nnodes, k)))

        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.sim = torch.zeros((nnodes, nnodes))

    def forward(self, x):
        with torch.no_grad():
            w = self.proto.data.clone()
            w = self.l2norm(w)
            self.proto.copy_(w)

        self.sim = F.cosine_similarity(
            self.proto.unsqueeze(0), self.proto.unsqueeze(1), dim=-1
        )
        self.sim = F.leaky_relu(self.sim)

        return self.sim
