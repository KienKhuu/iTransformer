import torch
from model.iTransformer import Model as iTransformer


class iTransformer_CL(iTransformer):
    def __init__(self, configs):
        super().__init__(configs)

    def encode(self, x_enc, x_mark_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev
        else:
            means, stdev = None, None

        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # B N E -> B N E
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # enc_out chính là 'h' (hidden representation)
        return enc_out, means, stdev

    def forecast_from_representation(self, h, means=None, stdev=None):
        N = means.shape[2] if means is not None else h.shape[1]

        # B N E -> B N S -> B S N
        dec_out = self.projector(h).permute(0, 2, 1)[:, :, :N]

        if self.use_norm and means is not None and stdev is not None:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out
