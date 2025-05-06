import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import get_mask_from_lengths, pad
from conformer import Conformer
from layers import SepConv1d
from symbols import symbols

def regulate_len(durations, enc_out, pace=1.0, max_dec_len=None):
    """If target=None, then predicted durations are applied"""
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0),
                               dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if max_dec_len is not None:
        enc_rep = enc_rep[:, :max_dec_len]
        dec_lens = torch.clip(dec_lens, max_dec_len)
    return enc_rep, dec_lens

def mas_width1(log_attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    neg_inf = log_attn_map.dtype.type(-np.inf)
    log_p = log_attn_map.copy()
    log_p[0, 1:] = neg_inf
    for i in range(1, log_p.shape[0]):
        prev_log1 = neg_inf
        for j in range(log_p.shape[1]):
            prev_log2 = log_p[i-1, j]
            log_p[i, j] += max(prev_log1, prev_log2)
            prev_log1 = prev_log2

    # now backtrack
    opt = np.zeros_like(log_p)
    one = opt.dtype.type(1)
    j = log_p.shape[1]-1
    for i in range(log_p.shape[0]-1, 0, -1):
        opt[i, j] = one
        if log_p[i-1, j-1] >= log_p[i-1, j]:
            j -= 1
            if j == 0:
                opt[1:i, j] = one
                break
    opt[0, j] = one
    return opt

class ConvAttention(torch.nn.Module):
    def __init__(self, n_sparc_channels=15, n_text_channels=256, n_att_channels=64):
        super(ConvAttention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.key_proj = nn.Sequential(
            SepConv1d(n_text_channels,
                     n_text_channels * 2,
                     kernel_size=3,
                     padding=1),
            torch.nn.SiLU(),
            SepConv1d(n_text_channels * 2,
                     n_att_channels,
                     kernel_size=1,)
        )

        self.query_proj = nn.Sequential(
            SepConv1d(n_sparc_channels,
                        n_sparc_channels * 2,
                        kernel_size=3,
                        padding=1),
            torch.nn.SiLU(),
            SepConv1d(n_sparc_channels * 2,
                        n_sparc_channels,
                        kernel_size=1,),
            torch.nn.SiLU(),
            SepConv1d(n_sparc_channels,
                        n_att_channels,
                        kernel_size=1,)
        )

    def forward(self, queries, keys, mask=None, attn_prior=None):
        """Attention mechanism for flowtron parallel
        Unlike in Flowtron, we have no restrictions such as causality etc,
        since we only need this during training.

        Args:
            queries (torch.tensor): B x T1 x C tensor (sparc data)
            keys (torch.tensor): B x T2 x C2 tensor (text data)
            mask (torch.tensor): B x T2 uint8 binary mask for variable length entries
                (should be in the T2 domain)
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask.
                Final dim T2 should sum to 1
        """
        keys_enc = self.key_proj(keys).permute(0,2,1)  # B x n_attn_dims x T2

        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = self.query_proj(queries).permute(0,2,1)

        # different ways of computing attn,
        # one is isotopic gaussians (per phoneme)
        # Simplistic Gaussian Isotopic Attention

        # B x n_attn_dims x T1 x T2
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2
        # compute log likelihood from a gaussian
        attn = -0.0005 * attn.sum(1, keepdim=True)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None]+1e-8)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # Softmax along T2
        return attn, attn_logprob

class TemporalPredictor(nn.Module):

    def __init__(self, input_size, filter_size, kernel_size, dropout):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(
            SepConv1d(input_size, filter_size, kernel_size, stride=1, padding=kernel_size//2),
            nn.SiLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(p=dropout),
            SepConv1d(filter_size, filter_size, kernel_size, stride=1, padding=kernel_size//2),
            nn.SiLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(p=dropout),
        )
        self.fc = nn.Linear(filter_size, 1)

    def forward(self, input, mask):
        out = input.masked_fill(mask, 0.)
        out = self.layers(out)
        out = self.fc(out).masked_fill(mask, 0.)
        return out.squeeze(-1)

class FastAccentTTS(nn.Module):
    def __init__(self, num_accents=2, sparc_dim=15, model_dim=256, encoder_n_layers=4, encoder_ff_dim=1024, encoder_kernel_size=9, encoder_ff_kernel_size=3, encoder_n_heads=2, 
                 n_aligner_channels=64,
                 predictor_filter_size=256, predictor_kernel_size=3, predictor_dropout=0.5,
                 decoder_n_layers=6, decoder_ff_dim=1024, decoder_kernel_size=9, decoder_ff_kernel_size=3, decoder_n_heads=2, dropout=0.2):
        super(FastAccentTTS, self).__init__()

        self.phone_embedding = nn.Embedding(len(symbols), model_dim)

        self.accent_embedding = nn.Embedding(num_accents, model_dim)

        self.phone_encoder = Conformer(n_layers=encoder_n_layers, d_model=model_dim, d_ff=encoder_ff_dim, 
                                kernel_size=encoder_kernel_size, ff_kernel_size=encoder_ff_kernel_size, n_head=encoder_n_heads, dropout=dropout)

        self.sparc_decoder = Conformer(decoder_n_layers, model_dim, decoder_ff_dim, decoder_kernel_size, decoder_ff_kernel_size, decoder_n_heads, dropout)

        self.attention = ConvAttention(n_text_channels=model_dim, n_sparc_channels=sparc_dim, n_att_channels=n_aligner_channels)

        self.duration_predictor = TemporalPredictor(model_dim, predictor_filter_size, predictor_kernel_size, predictor_dropout)

        self.sparc_proj = nn.Linear(model_dim, sparc_dim)

    def _binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_out_cpu = np.zeros(attn.data.shape, dtype=np.float32)
            log_attn_cpu = torch.log(attn.data).to(device='cpu', dtype=torch.float32)
            log_attn_cpu = log_attn_cpu.numpy()
            out_lens_cpu = out_lens.cpu()
            in_lens_cpu = in_lens.cpu()
            for ind in range(b_size):
                hard_attn = mas_width1(
                    log_attn_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]])
                attn_out_cpu[ind, 0, :out_lens_cpu[ind], :in_lens_cpu[ind]] = hard_attn
            attn_out = torch.tensor(
                attn_out_cpu, device=attn.get_device(), dtype=attn.dtype)
        return attn_out

    def forward(self, phones, phone_lens, max_phone_len, attn_prior, sparc, target_lens, max_target_len, accent_label):
        # phones -> masked -> embeddings
        enc_mask = get_mask_from_lengths(phone_lens, max_phone_len)
        
        embs = self.phone_embedding(phones)

        accent_emb = self.accent_embedding(accent_label)

        embs += accent_emb
        # embeddings -> encoded_embs
        enc_out = self.phone_encoder(embs, key_padding_mask=enc_mask)
        # regulate_lengths(encoded_embs, durations) -> expanded_embs
        # expanded_embs -> decoded_embs
        log_dur_pred = self.duration_predictor(enc_out, enc_mask)
        # dur_pred = torch.clip(torch.exp(log_dur_pred) - 1, 0)

        attn_soft, attn_logprob = self.attention(
            sparc, embs.permute(0, 2, 1), enc_mask, attn_prior=attn_prior)
        
        attn_hard = self._binarize_attention(attn_soft, phone_lens, target_lens)

        # Viterbi --> durations
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        dur_tgt = attn_hard_dur
        assert torch.all(torch.eq(dur_tgt.sum(dim=1), target_lens))

        expanded_embs, _ = regulate_len(enc_out, dur_tgt, max_dec_len=max_target_len)
        sparc_masks = get_mask_from_lengths(target_lens, max_target_len)
        decoded_embs = self.sparc_decoder(expanded_embs, key_padding_mask=sparc_masks)
        # decoded_embs -> pred_sparc
        pred_sparc = self.sparc_proj(decoded_embs)
        return (
            log_dur_pred,
            dur_tgt,
            phone_lens,
            enc_mask,
            pred_sparc, 
            target_lens,
            sparc_masks,
            attn_soft,
            attn_hard,
            attn_logprob,
        )
    
    # def inference(self, phones, phone_lens, max_phone_len):
    #     # phones -> masked -> embeddings
    #     phone_masks = get_mask_from_lengths(phone_lens, max_phone_len)
        
    #     phone_embs = self.phone_embedding(phones)
    #     # embeddings -> encoded_embs
    #     encoded_embs = self.phone_encoder(phone_embs, key_padding_mask=phone_masks)
    #     # regulate_lengths(encoded_embs, durations) -> expanded_embs
    #     # expanded_embs -> decoded_embs
    #     expanded_embs, _ = regulate_len(encoded_embs, durations, max_dec_len=max_target_len)
    #     sparc_masks = get_mask_from_lengths(target_lens, max_target_len)
    #     decoded_embs = self.sparc_decoder(expanded_embs, key_padding_mask=sparc_masks)
    #     # decoded_embs -> pred_sparc
    #     pred_sparc = self.sparc_proj(decoded_embs)
    #     return (
    #         pred_sparc, 
    #         sparc_masks,
    #     )