import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Embed import DataEmbedding_inverted
import numpy as np


class FrequencyDecompositionLayer(nn.Module):
    def __init__(self, cutoff_ratio=0.1, learnable_alpha=False, taper_ratio=0.1):
        super(FrequencyDecompositionLayer, self).__init__()
        self.taper_ratio = taper_ratio  # fraction of spectrum used as the smooth transition band
        self.learnable_alpha = learnable_alpha

        if learnable_alpha:
            # Parameterize alpha in logit space so a sigmoid maps it back to (0, 1)
            # while leaving it as a free, gradient-trainable parameter.
            cutoff_ratio = min(max(cutoff_ratio, 1e-4), 1 - 1e-4)  # keep logit finite
            init_logit = torch.log(torch.tensor(cutoff_ratio / (1 - cutoff_ratio)))
            self.alpha_logit = nn.Parameter(init_logit)
        else:
            # Fixed cutoff ratio, stored as a buffer so it is saved/loaded with the
            # model checkpoint and can be read back for reporting.
            self.register_buffer('cutoff_ratio', torch.tensor(float(cutoff_ratio)))

    def get_alpha(self):
        """Returns the current cutoff ratio alpha as a scalar tensor (differentiable if learnable)."""
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logit)
        return self.cutoff_ratio

    def forward(self, x):
        """
        x: [B, L, D]
        Returns:
            trend: [B, L, D]
            seasonality: [B, L, D]
        """
        B, L, D = x.shape

        # FFT along temporal dimension
        x_fft = torch.fft.rfft(x, dim=1)
        num_bins = x_fft.size(1)

        alpha = self.get_alpha()
        cutoff = alpha * num_bins  # continuous cutoff position (differentiable if alpha is learnable)

        # Raised-cosine (Hann-style) tapered low-pass mask: transitions smoothly
        # from ~1 to ~0 over a band of width `taper_width` centered on `cutoff`,
        # instead of a hard step, to suppress spectral leakage / Gibbs ringing.
        freq_idx = torch.arange(num_bins, device=x.device, dtype=x.dtype)
        taper_width = max(1.0, self.taper_ratio * num_bins)
        mask_1d = torch.sigmoid(-(freq_idx - cutoff) / (taper_width / 4.0 + 1e-6))
        mask = mask_1d.view(1, num_bins, 1).to(x_fft.dtype)  # broadcast over batch & channel dims

        # Apply tapered low-pass filter
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

        # Frequency-aware Decomposition Layer.
        # alpha / learnable_alpha / taper_ratio are read from configs (with safe
        # defaults) so per-dataset values are explicit, loggable, and sweepable
        # without editing this file — see class docstring above for rationale.
        alpha = getattr(configs, 'alpha', 0.1)
        learnable_alpha = bool(getattr(configs, 'learnable_alpha', False))
        taper_ratio = getattr(configs, 'taper_ratio', 0.1)
        self.decomposition = FrequencyDecompositionLayer(
            cutoff_ratio=alpha,
            learnable_alpha=learnable_alpha,
            taper_ratio=taper_ratio
        )

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
                        FullAttention(
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
            # Trend forecasting head: learns to extend the trend from seq_len -> pred_len
            # instead of assuming the historical trend repeats (naive slicing).
            self.trend_projection = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
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

        # Forecast the trend forward via learned projection (seq_len -> pred_len)
        # trend: [B, seq_len, D] -> permute to [B, D, seq_len] -> linear -> [B, D, pred_len] -> back to [B, pred_len, D]
        trend_pred = self.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = dec_out + trend_pred

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def get_current_alpha(self):
        """
        Utility for logging: returns the current alpha (cutoff ratio) as a
        Python float. Call this after training to report the alpha actually
        used/learned for a given dataset (addresses R2/R4: per-dataset alpha
        reporting).
        """
        return float(self.decomposition.get_alpha().detach().cpu())
