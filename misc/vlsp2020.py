#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert VLSP2020 set to huggingface audio datasets format"""

###############################################################################
# commands to download files in colab

# !gdown 1vUSxdORDxk-ePUt-bUVDahpoXiqKchMx
# !tar -xvf 'VinBigdata-VLSP2020-100h (1).rar'  # SOMEHOW not rar file but tar file
# !rm 'VinBigdata-VLSP2020-100h (1).rar'
# !mkdir data
# %pip install -qU 'datasets[audio]'

###############################################################################

import os, pandas as pd
from glob import glob
from tqdm import tqdm

folder = "vlsp2020_train_set_02/"
list_txt_files = glob("*.txt", root_dir=folder)
list_wav_files = glob("*.wav", root_dir=folder)

assert set(i.replace(".wav", "") for i in list_wav_files) - set(i.replace(".txt", "") for i in list_txt_files) == set()
assert set(i.replace(".txt", "") for i in list_txt_files) - set(i.replace(".wav", "") for i in list_wav_files) != set()  # ATTENTION

rows_list = []
for row in tqdm(list_wav_files):  # cannot use list_txt_files see assertion above
	os.rename(folder + row, "data/" + row)  # re-organize
	with open(folder + row.replace(".wav", ".txt"), mode="r", encoding="utf8") as f:
		txt = f.read().replace("<unk>", "").replace("  ", " ")
	rows_list.append({"file_name": row, "transcription": txt})

pd.DataFrame(rows_list).to_csv("data/metadata.csv", index=False, quoting=1)  # formatted following huggingface docs

###############################################################################

from datasets import load_dataset
dataset = load_dataset("audiofolder", data_dir="data")
dataset.push_to_hub("doof-ferb/vlsp2020_vinai_100h", token="███")
