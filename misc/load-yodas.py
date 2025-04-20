#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""extract vietnamese subset of the YODAS dataset"""

# %pip install -qU 'datasets[audio]'
import datasets as hugDS

ds = (hugDS
	.load_dataset("espnet/yodas", name="vi000", split="train", trust_remote_code=True)  # ~100 GB
	.remove_columns(["id", "utt_id"])
	.rename_column("text", "transcription")
)

(hugDS
	.Dataset.from_generator(ds.__iter__)  # a hack from https://github.com/huggingface/datasets/issues/5665
	.cast_column("audio", hugDS.Audio(sampling_rate=16_000))  # take back column type
	.push_to_hub("doof-ferb/YODAS_vie", token="███")
)
