import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FuzzyAttention
from layers.Embed import DataEmbedding_inverted
import numpy as np


class FrequencyDecompositionLayer(nn.Module):
    """
    Frequency-aware decomposition using FFT-based low-pass filtering
    """
    def __init__(self, cutoff_ratio=0.1):
        super(FrequencyDecompositionLayer, self).__init__()
        self.cutoff_ratio = cutoff_ratio  # fraction of low frequencies to keep

    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, D = x.shape

        # FFT along temporal dimension
        x_fft = torch.fft.rfft(x, dim=1)

        # Determine cutoff index
        cutoff = int(self.cutoff_ratio * x_fft.size(1))

        # Low-pass mask
        mask = torch.zeros_like(x_fft)
        mask[:, :cutoff, :] = 1.0

        # Apply low-pass filter
        trend_fft = x_fft * mask

        # Inverse FFT to get trend
        trend = torch.fft.irfft(trend_fft, n=L, dim=1)

        # Seasonal component as residual
        seasonality = x - trend

        return trend, seasonality


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Frequency-aware Decomposition Layer
        self.decomposition = FrequencyDecompositionLayer(cutoff_ratio=0.1)

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FuzzyAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Projection head
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Frequency-aware Decomposition
        trend, seasonality = self.decomposition(x_enc)

        # Normalization (Non-stationary Transformer style)
        means = seasonality.mean(1, keepdim=True).detach()
        seasonality = seasonality - means
        stdev = torch.sqrt(
            torch.var(seasonality, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        seasonality = seasonality / stdev

        _, _, N = seasonality.shape

        # Embedding
        enc_out = self.enc_embedding(seasonality, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1)

        # Add trend back
        dec_out = dec_out + trend[:, -self.pred_len:, :]

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
