#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BE AWARE: the huggingface docker image use python 3.8

"""download data to local cache for training + evaluation"""

from datasets import load_dataset

# _ = load_dataset(path="mozilla-foundation/common_voice_16_1", name="vi", trust_remote_code=True, split="test")
# _ = load_dataset(path="google/fleurs", name="vi_vn", trust_remote_code=True, split="test")
# _ = load_dataset(path="vivos", trust_remote_code=True, split="test")
# _ = load_dataset(path="linhtran92/viet_bud500", split="test")
_ = load_dataset(path="doof-ferb/infore1_25hours", split="train")
_ = load_dataset(path="doof-ferb/fpt_fosd", split="train")
_ = load_dataset(path="doof-ferb/LSVSC")
