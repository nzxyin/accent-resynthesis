import torch
import torch.nn as nn
from util import get_mask_from_lengths, pad
from conformer import Conformer
from symbols import symbols

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        sparc_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            sparc_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(sparc_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, sparc_len = self.LR(x, duration, max_len)
        return output, sparc_len

class AccentResynthesisTTS(nn.Module):
    def __init__(self, sparc_dim=15, model_dim=256, encoder_n_layers=4, encoder_ff_dim=1024, encoder_kernel_size=9, encoder_ff_kernel_size=3, encoder_n_heads=2, 
                 decoder_n_layers=6, decoder_ff_dim=1024, decoder_kernel_size=9, decoder_ff_kernel_size=3, decoder_n_heads=2, dropout=0.2):
        super(AccentResynthesisTTS, self).__init__()

        self.phone_embedding = nn.Embedding(len(symbols), model_dim)

        self.phone_encoder = Conformer(n_layers=encoder_n_layers, d_model=model_dim, d_ff=encoder_ff_dim, 
                                kernel_size=encoder_kernel_size, ff_kernel_size=encoder_ff_kernel_size, n_head=encoder_n_heads, dropout=dropout)

        self.sparc_decoder = Conformer(decoder_n_layers, model_dim, decoder_ff_dim, decoder_kernel_size, decoder_ff_kernel_size, decoder_n_heads, dropout)

        self.lr = LengthRegulator()

        self.sparc_proj = nn.Linear(model_dim, sparc_dim)

    def forward(self, phones, phone_lens, max_phone_len, durations, target_lens, max_target_len):
        # phones -> masked -> embeddings
        phone_masks = get_mask_from_lengths(phone_lens, max_phone_len)
        
        phone_embs = self.phone_embedding(phones)
        # embeddings -> encoded_embs
        encoded_embs = self.phone_encoder(phone_embs, key_padding_mask=phone_masks)
        # regulate_lengths(encoded_embs, durations) -> expanded_embs
        # expanded_embs -> decoded_embs
        expanded_embs, _ = self.lr(encoded_embs, durations, max_target_len)
        sparc_masks = get_mask_from_lengths(target_lens, max_target_len)
        decoded_embs = self.sparc_decoder(expanded_embs, key_padding_mask=sparc_masks)
        # decoded_embs -> pred_sparc
        pred_sparc = self.sparc_proj(decoded_embs)
        return (
            pred_sparc, 
            sparc_masks,
        )