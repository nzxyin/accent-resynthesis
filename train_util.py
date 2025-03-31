from model import AccentResynthesisTTS
from optimizer import ScheduledOptim

import os
import json

import torch
import numpy as np

def to_device(data, device):
    assert len(data) == 8, "incorrect data format"
    (
        ids,
        texts,
        text_lens,
        max_text_len,
        durations,
        sparc_lens,
        max_sparc_len,
        sparcs,
    ) = data
    texts = torch.from_numpy(texts).long().to(device)
    text_lens = torch.from_numpy(text_lens).to(device)
    sparcs = torch.from_numpy(sparcs).float().to(device)
    sparc_lens = torch.from_numpy(sparc_lens).to(device)
    durations = torch.from_numpy(durations).long().to(device)
    return (
        ids,
        texts,
        text_lens,
        max_text_len,
        durations,
        sparc_lens,
        max_sparc_len,
        sparcs,
    )

def get_model(args, configs, device, train=False):
    (_, model_config, train_config) = configs

    model = AccentResynthesisTTS(**model_config).to(device)
    if args.load_path:
        ckpt_path = args.load_path
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        msg = model.load_state_dict(ckpt["model"])
        print(msg)
    
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, weights_only=False)
        print(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def log(logger, step, loss):
    logger.add_scalar("Loss/reconstruction_loss", loss, step)