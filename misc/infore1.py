#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert infore1 (25h) set to huggingface audio datasets format"""

###############################################################################
# commands to download files in colab

# !wget https://files.huylenguyen.com/datasets/infore/25hours.zip
# !unzip -P BroughtToYouByInfoRe 25hours.zip
# !rm 25hours.zip
# !mkdir data
# %pip install -qU 'datasets[audio]'

###############################################################################

import os, pandas as pd
from glob import glob
from tqdm import tqdm

# re-organize files following huggingface docs
for path in tqdm(glob("**/*.wav", root_dir="25hours/wavs", recursive=True)):
	os.rename("25hours/wavs/" + path, "data/" + path.split("/")[-1])

df = pd.read_csv("25hours/scripts.csv", sep="|", names=["file_name", "transcription"], usecols=[0,1])
df["file_name"] = df["file_name"].str.split("/").str[-1]
df.to_csv("data/metadata.csv", index=False, quoting=1)  # formatted following huggingface docs

###############################################################################

from datasets import load_dataset
dataset = load_dataset("audiofolder", data_dir="data")
dataset.push_to_hub("doof-ferb/infore1_25hours", token="███")
