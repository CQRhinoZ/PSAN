import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

##########################

## Spatio-temporal Encoder

##########################

class STEncoder(nn.Module):
    def __init__(self, in_len, k, nnodes, emb_dim, layer, nheads):
        super().__init__()
        self.emb_dim = emb_dim
        self.layer = layer

        self.T = nn.ModuleList([
            BiLSTCM(in_len, emb_dim) for _ in range(layer)
        ])
    
        self.S = nn.ModuleList([
            SpatialMHSA(emb_dim, nheads) for _ in range(layer)
        ])
    
        self.resi = nn.ModuleList([
            FeedForwardNet(emb_dim) for _ in range(layer)
        ])
    
        self.pn = PLM(emb_dim, nnodes, k=k)
    
    def forward(self, x):
        """
        :param x: BxTxNxD
        :return: BxTxNxD
        """
        sim = self.pn(x)
        for i in range(self.layer):
            residual = x
            x = self.T[i](x)
            x = torch.add(x, residual)
            residual = x
            x = self.S[i](x, sim=sim)
            x = self.resi[i](x, residual)
        return x

class Net(nn.Module):
    def __init__(
            self,
            k,
            nnodes,
            in_dim,
            out_dim,
            layers,
            nheads,
            window=24,
    ):
        super().__init__()
        self.in_len = window
        self.out_len = window
        self.k = k
        self.nnodes = nnodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = layers
        self.nheads = nheads
        self.emb_dim = 64

        self.DataEmb = DataEmbedding(self.emb_dim, self.in_len, self.out_len, nnodes)
    
        self.Encoder = STEncoder(self.in_len, self.k, nnodes, self.emb_dim, layers, nheads)
    
        self.imputer = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.out_dim)
        )
    
    def process_before(self, x, u, mask):
        x = x * mask
        u = u.unsqueeze(dim=2).expand(-1,-1,x.shape[2], -1)
        x = torch.cat([x, u], dim=-1)
        return x
    
    def forward(self, x, u, mask):
        """
        :param x: BxTxNx1
        :return: Bx1xNxT
        """
        x = self.process_before(x, u, mask)
        x = self.DataEmb(x)  # BxTxNxD
        out = self.Encoder(x)
        out = self.imputer(out)  # BxTxNx1
        return out
    
    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--k', type=int, default=10)
        parser.add_argument('--nnodes', type=int, default=207)
        parser.add_argument('--in-dim', type=int, default=3)
        parser.add_argument('--out-dim', type=int, default=1)
        parser.add_argument('--layers', type=int, default=3)
        parser.add_argument('--nheads', type=int, default=4)
        return parser
