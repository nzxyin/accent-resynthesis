import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from util import pad_1D, pad_2D


class SingleAccentDataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.splits_path = preprocess_config["path"]["splits_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.basename = self.process_meta(filename)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.basename)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        phone = np.load(os.path.join(self.preprocessed_path, "text_ids", f"{basename}.npy"))
        sparc = np.load(os.path.join(self.preprocessed_path, "sparc", f"{basename}.npy"))
        duration = np.load(os.path.join(self.preprocessed_path, "durations", f"{basename}.npy"))

        sample = {
            "id": basename,
            "text": phone,
            "sparc": sparc,
            "duration": duration,
        }
        return sample

    def process_meta(self, filename):
        with open(os.path.join(self.splits_path, filename), "r") as f:
            name = json.load(f)
        return name

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        sparcs = [data[idx]["sparc"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        sparc_lens = np.array([sparc.shape[0] for sparc in sparcs])

        texts = pad_1D(texts)
        sparcs = pad_2D(sparcs)
        durations = pad_1D(durations)

        assert texts.shape[1] == max(text_lens)

        return (
            ids,
            texts,
            text_lens,
            max(text_lens),
            durations,
            sparc_lens,
            max(sparc_lens),
            sparcs,
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