import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.d_model = configs.d_model
        self.representation_dim = configs.d_model

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.class_strategy = configs.class_strategy
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def _encode_tokens(self, x_enc, x_mark_enc=None):
        """Return iTransformer tokens before the forecasting projector.

        Returns:
            enc_out: [B, token_count, d_model]
            attns: attention maps
            N: number of original input variates (excludes time-covariate tokens)
            means/stdev: normalization statistics for forecast de-normalization
        """
        means = stdev = None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out, attns, N, means, stdev

    def encode(self, x_enc, x_mark_enc=None, mask=None):
        """Global latent representation used by contrastive pretraining: [B, d_model]."""
        enc_out, _, N, _, _ = self._encode_tokens(x_enc, x_mark_enc)
        # DataEmbedding_inverted may append time-covariate tokens. Exclude them.
        return enc_out[:, :N, :].mean(dim=1)

    def forecast(self, x_enc, x_mark_enc=None):
        enc_out, attns, N, means, stdev = self._encode_tokens(x_enc, x_mark_enc)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc)
        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        return dec_out[:, -self.pred_len :, :]
