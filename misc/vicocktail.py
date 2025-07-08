#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert ViCocktail set to huggingface video datasets format"""

###############################################################################

# %pip install -qU 'huggingface_hub[hf_xet]' 'datasets[audio,vision]' av

###############################################################################

import os, datasets as hugDS
from tqdm import tqdm

ds = hugDS.load_dataset("nguyenvulebinh/ViCocktail") # may take up to 210 GiB after unpacking

# huggingface datasets version ≥ 3.4
for split, data in ds.items():
	os.makedirs(split, exist_ok=True)

	with open(split+"/metadata.csv", "w", encoding="utf-8") as txt_file:
		txt_file.write('"file_name","sample_id","text"\n')

		for sample in tqdm(data):
			vid_path = sample["__key__"] + ".mp4"
			sample_id = str(sample["sample_id"], encoding="utf-8")
			txt = str(sample["label"], encoding="utf-8").replace('"','')
			txt_file.write(f'"{vid_path}","{sample_id}","{txt}"\n')

			with open(split+"/"+vid_path, "wb") as vid_file:
				vid_file.write(sample["video"])

###############################################################################

from os import listdir
from datasets import DatasetDict, load_dataset

ds = DatasetDict()
for d in listdir():
	ds[d] = load_dataset("videofolder", data_dir=d, split="train")

## don’t do this
# dataset = load_dataset("videofolder", data_dir=".")
## failed to recognize custom named splits other than train/test/dev

ds.push_to_hub("doof-ferb/ViCocktail", token="███")
