import json
import math
import os

import numpy as np
from torch.utils.data import Dataset
import functools
from scipy import ndimage
from scipy.stats import betabinom

from util import pad_1D, pad_2D

class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_sparc_len_to=100, round_text_len_to=20):
        self.round_sparc_len_to = round_sparc_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_sparc_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, sparc_count, scaling=1.0):
    P = phoneme_count
    M = sparc_count
    x = np.arange(0, P)
    sparc_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        sparc_i_prob = rv.pmf(x)
        sparc_text_probs.append(sparc_i_prob)
    return np.array(sparc_text_probs)


def extract_duration(text_len, sparc_len):
    binomial_interpolator = BetaBinomialInterpolator()
    attn_prior = binomial_interpolator(sparc_len, text_len)
    assert sparc_len == attn_prior.shape[0]
    return attn_prior


class MultiAccentDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.splits_path = preprocess_config["path"]["splits_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.name_accent = self.process_meta(filename)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename, accent = self.name_accent[idx]
        phone = np.load(os.path.join(self.preprocessed_path, "text_ids", f"{basename}.npy"))
        sparc = np.load(os.path.join(self.preprocessed_path, "sparc", f"{basename}.npy"))
        duration = np.load(os.path.join(self.preprocessed_path, "duration", f"{basename}.npy"))

        sample = {
            "id": basename,
            "text": phone,
            "sparc": sparc,
            "duration": duration,
            "accent": accent,
        }
        return sample

    def process_meta(self, filename):
        with open(os.path.join(self.splits_path, filename), "r") as f:
            name_accent = [(name, accent) for name, accent in json.load(f).items()]
        return name_accent

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        sparcs = [data[idx]["sparc"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        sparc_lens = np.array([sparc.shape[0] for sparc in sparcs])
        accents = np.array([data[idx]["accent"] for idx in idxs])

        texts = pad_1D(texts)
        sparcs = pad_2D(sparcs)
        
        durs_padded = np.zeros((len(idxs), max(sparc_lens), max(text_lens)))
        for i, dur in enumerate(durations):
            durs_padded[i, :dur.shape[0], :dur.shape[1]] = dur

        assert texts.shape[1] == max(text_lens)
        assert sparcs.shape[1] == max(sparc_lens)
        assert (durs_padded.shape[1], durs_padded.shape[2]) == (max(sparc_lens), max(text_lens))

        return (
            ids,
            texts,
            text_lens,
            max(text_lens),
            durs_padded,
            sparc_lens,
            max(sparc_lens),
            sparcs,
            accents,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output