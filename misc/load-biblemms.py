#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract vietnamese subset of the BibleMMS dataset
"""

# %pip install -qU 'datasets[audio]'
import datasets as hugDS

ds = (hugDS
	.load_dataset("Flux9665/BibleMMS", streaming=True, split="train")
	.filter(lambda lang: lang == "vie", input_columns=["language_code"])
	.remove_columns("language_code")
)

(hugDS
	.Dataset.from_generator(ds.__iter__)  # a hack from https://github.com/huggingface/datasets/issues/5665
	.cast_column("audio", hugDS.Audio(sampling_rate=16_000))  # take back column type
	.push_to_hub("doof-ferb/BibleMMS_vie", token="███")
)
