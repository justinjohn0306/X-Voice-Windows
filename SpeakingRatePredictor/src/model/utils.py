from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style
import math
import re
from typing import List

def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# helpers

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers

def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax() # 返回张量t中最大的值

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

# filter func for dirty data with many repetitions

def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False