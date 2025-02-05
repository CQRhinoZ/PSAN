import torch
from torch import nn

from .rnn_imputers import BiRNNImputer
from tsl.utils.parser_utils import str_to_bool
from tsl.nn.functional import reverse_tensor


class Generator(nn.Module):
    def __init__(self, d_in, d_model, d_z, dropout=0., inject_noise=True):
        super(Generator, self).__init__()
        self.inject_noise = inject_noise
        self.d_z = d_z if inject_noise else 0
        self.birnn = BiRNNImputer(d_in,
                                  d_model,
                                  d_u=d_z,
                                  concat_mask=True,
                                  detach_inputs=False,
                                  dropout=dropout,
                                  state_init='zero')

    def forward(self, x, mask):
        if len(x.shape) == 4:
            x = x.squeeze(dim=-1)
            mask = mask.squeeze(dim=-1)
        if self.inject_noise:
            z = torch.rand(x.size(0), x.size(1), self.d_z, device=x.device) * 0.1
        else:
            z = None
        return self.birnn(x, mask, u=z)


class Discriminator(torch.nn.Module):
    def __init__(self, d_in, d_model, dropout=0.):
        super(Discriminator, self).__init__()
        self.birnn = nn.GRU(2 * d_in, d_model, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.read_out = nn.Linear(2 * d_model, d_in)

    def forward(self, x, h):
        if len(x.shape) == 4:
            x = x.squeeze(dim=-1)
            h = h.squeeze(dim=-1)
        x_in = torch.cat([x, h], dim=-1)
        out, _ = self.birnn(x_in)
        logits = self.read_out(self.dropout(out))
        if len(logits.shape) == 3:
            logits = logits.unsqueeze(dim=-1)
        return logits

class RGAINNet(torch.nn.Module):
    def __init__(self, d_in, d_model, d_z, dropout=0., inject_noise=False, k=5):
        super(RGAINNet, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_z = d_z
        self.inject_noise = inject_noise
        self.k = k
        self.generator = Generator(d_in, d_model, d_z=d_z, dropout=dropout, inject_noise=inject_noise)
        self.discriminator = Discriminator(d_in, d_model, dropout)

    def forward(self, x, mask, **kwargs):
        x = x.squeeze(dim=-1)
        mask = mask.squeeze(dim=-1)
        # print(self.d_in, self.d_model, self.d_z, self.inject_noise, self.k)
        if not self.training and self.inject_noise:
            res = []
            for _ in range(self.k):
                res.append(self.generator(x, mask)[0])
            return torch.stack(res, 0).mean(0),

        return self.generator(x, mask)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int)
        parser.add_argument('--d-model', type=int, default=None)
        parser.add_argument('--d-z', type=int, default=8)
        parser.add_argument('--k', type=int, default=5)
        parser.add_argument('--inject-noise', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--dropout', type=float, default=0.)
        return parser
