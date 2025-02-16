import torch
import torch.nn as nn
from ..core import modules


class EncoderLayerForSAnD(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2) -> None:
        super(EncoderLayerForSAnD, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)

        return x


class SAnD(nn.Module):
    """
    Simply Attend and Diagnose model

    The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)

    `Attend and Diagnose: Clinical Time Series Analysis Using Attention Models <https://arxiv.org/abs/1711.03905>`_
    Huan Song, Deepta Rajan, Jayaraman J. Thiagarajan, Andreas Spanias
    """
    def __init__(
            self, input_features: int, seq_len: int, n_heads: int, factor: int,
            n_class: int, n_layers: int, d_model: int = 128, dropout_rate: float = 0.2

    ) -> None:
        super(SAnD, self).__init__()
        self.encoder = EncoderLayerForSAnD(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.dense_interpolation(x)
        x = self.clf(x)
        return x

class TransformerSiamesReeflexSoH(nn.Module):
    def __init__(self, feats):
        super(TransformerSiamesReeflexSoH, self).__init__()
        self.name = 'TransformerSiamesReeflexSoH'
        self.lr = 0.004
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.margin = 1
        self.n = self.n_feats * self.n_window

        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)

        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Hardsigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward_once(self, src, tgt):
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder(*self.encode(src, c, tgt)))

        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder(*self.encode(src, c, tgt)))

        return x2

    def forward(self, src1, tgt1, src2, tgt2):
        x1 = self.forward_once(src1, tgt1)
        x2 = self.forward_once(src2, tgt2)

        return x1, x2




