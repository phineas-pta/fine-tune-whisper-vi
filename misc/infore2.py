#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert infore2 (audiobooks) set to huggingface audio datasets format"""

###############################################################################
# commands to download files in colab
# zip file + extracted contents eat up all colab disk space so instead stream zip file to bsdtar to extract on-the-fly

# !apt install libarchive-tools
# !wget -O - https://files.huylenguyen.com/datasets/infore/audiobooks.zip | bsdtar --passphrase BroughtToYouByInfoRe -xf - -C .
# !mkdir data
# %pip install -qU 'datasets[audio]'

###############################################################################

import os, json, pandas as pd
from glob import glob
from tqdm import tqdm

# re-organize files following huggingface docs
for path in tqdm(glob("**/*.wav", root_dir="book_relocated", recursive=True)):
	os.rename("book_relocated/" + path, "data/" + path.split("/")[-1])

# badly formatted json so pandas.read_json fail, use json.loads to process
with open("data_book_train_relocated.json", mode="r", encoding="utf8") as f:
	raw_df = f.read().splitlines()

df = (pd
	.DataFrame([json.loads(i) for i in raw_df])
	.rename(columns={"key": "file_name", "text": "transcription"})
)[["file_name", "transcription"]]
df["file_name"] = df["file_name"].str.split("/").str[-1]
df.to_csv("data/metadata.csv", index=False, quoting=1)  # formatted following huggingface docs

###############################################################################

from datasets import load_dataset
dataset = load_dataset("audiofolder", data_dir="data")
dataset.push_to_hub("doof-ferb/infore2_audiobooks", token="███")
