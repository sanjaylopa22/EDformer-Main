import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Embed import DataEmbedding_inverted


class DecompositionLayer(nn.Module):
    """
    Moving-average based decomposition. Separates a multivariate time
    series into trend (smoothed via average pooling) and seasonal
    (residual) components.
    """
    def __init__(self, kernel_size):
        super(DecompositionLayer, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        """
        x: [B, L, D]
        Returns:
            trend: [B, L, D]
            seasonality: [B, L, D]
        """
        trend = self.moving_avg(x.permute(0, 2, 1))  # Apply moving average to get trend
        trend = trend.permute(0, 2, 1)                # Permute back to original shape
        seasonality = x - trend                        # Residual component
        return trend, seasonality


class Model(nn.Module):
    """
    EDformer with moving-average decomposition (forecasting-only variant).

    Pipeline:
      1. Decompose input into trend (moving average) and seasonal
         (residual) components.
      2. Normalize and embed the seasonal component (variate-as-token,
         following the inverted embedding scheme).
      3. Encode with multivariate self-attention (FullAttention) + FFN.
      4. Project seasonal representation to the forecast horizon and
         de-normalize.
      5. Forecast the trend forward via a learned linear projection
         (seq_len -> pred_len), rather than assuming the historical trend
         is locally constant/repeating.
      6. Sum seasonal forecast + trend forecast.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        assert self.task_name in ['long_term_forecast', 'short_term_forecast'], \
            f"EDformer (forecasting-only) does not support task_name='{self.task_name}'"

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomposition Layer
        self.decomposition = DecompositionLayer(kernel_size=25)

        # Embedding (seasonal component, variate-as-token)
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
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Seasonal projection head: d_model -> pred_len
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Trend forecasting head: learns to extend the trend from
        # seq_len -> pred_len instead of assuming the historical trend
        # repeats (naive slicing of trend[:, -pred_len:, :]).
        self.trend_projection = nn.Linear(configs.seq_len, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Decomposition
        trend, seasonality = self.decomposition(x_enc)

        # Normalization from Non-stationary Transformer, applied to the
        # seasonal component only
        means = seasonality.mean(1, keepdim=True).detach()
        seasonality = seasonality - means
        stdev = torch.sqrt(torch.var(seasonality, dim=1, keepdim=True, unbiased=False) + 1e-5)
        seasonality = seasonality / stdev

        _, _, N = seasonality.shape

        # Embedding
        enc_out = self.enc_embedding(seasonality, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Seasonal projection to forecast horizon
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Forecast the trend forward via learned projection (seq_len -> pred_len)
        # trend: [B, seq_len, D] -> [B, D, seq_len] -> Linear -> [B, D, pred_len] -> [B, pred_len, D]
        trend_pred = self.trend_projection(trend.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = dec_out + trend_pred

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, D]
