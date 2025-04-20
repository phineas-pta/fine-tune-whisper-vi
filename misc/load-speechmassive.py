#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""extract vietnamese subset of the Speech-MASSIVE dataset"""

# %pip install -qU 'datasets[audio]'
import datasets as hugDS

# columns to be removed
cols = ["locale", "partition", "scenario", "intent_idx", "slot_method", "judgments", "tokens", "labels", "annot_utt", "path"]

ds_train = hugDS.load_dataset("FBK-MT/Speech-MASSIVE",      streaming=True, name="vi-VN", split="train_115" ).remove_columns(cols)
ds_valid = hugDS.load_dataset("FBK-MT/Speech-MASSIVE",      streaming=True, name="vi-VN", split="validation").remove_columns(cols)
ds_test  = hugDS.load_dataset("FBK-MT/Speech-MASSIVE-test", streaming=True, name="vi-VN", split="test"      ).remove_columns(cols)

ds_vi = hugDS.DatasetDict()
# a hack from https://github.com/huggingface/datasets/issues/5665
ds_vi["train"]      = hugDS.Dataset.from_generator(ds_train.__iter__)
ds_vi["validation"] = hugDS.Dataset.from_generator(ds_valid.__iter__)
ds_vi["test"]       = hugDS.Dataset.from_generator(ds_test.__iter__)
ds_vi = ds_vi.cast_column("audio", hugDS.Audio(sampling_rate=16_000))  # take back column type

ds_vi.push_to_hub("doof-ferb/Speech-MASSIVE_vie", token="███")
