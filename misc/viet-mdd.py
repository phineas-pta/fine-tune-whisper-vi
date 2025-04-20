#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert VietMDD set to huggingface audio datasets format"""

###############################################################################
# commands to download files in colab

# !gdown --folder 1TjTluTxEB99QhGFTYFWb-vEdWXM-lyKJ  # doesn’t work because 50 files limit
# may try this one but very slow
# !wget https://gist.githubusercontent.com/fish4terrisa-MSDSM/1feb50c1370f6463357a9639381416f4/raw/05221e4f20f0fb7f099fd6d9e3cc299078ff18e1/gdown_folder.py
# !python gdown_folder.py 1TjTluTxEB99QhGFTYFWb-vEdWXM-lyKJ
# or process locally (~600 MB data)

# download files list for train/validation/test sets
# !wget https://raw.githubusercontent.com/VietMDDDataset/VietMDD/refs/heads/main/train.csv
# !wget https://raw.githubusercontent.com/VietMDDDataset/VietMDD/refs/heads/main/dev.csv
# !wget https://raw.githubusercontent.com/VietMDDDataset/VietMDD/refs/heads/main/test_fix.csv
# !mkdir -p vietmdd/{train,validation,test,orphan}

# %pip install -qU 'datasets[audio]'

###############################################################################

import os, pandas as pd
from glob import glob
from tqdm import tqdm

# retrieve files list & transcription
full_df = pd.read_csv("audio_MDD/Data_MDD_text.csv", index_col=0)
train_files = pd.read_csv("train.csv",    index_col=0)["Path"]
valid_files = pd.read_csv("dev.csv",      index_col=0)["Path"]
test_files  = pd.read_csv("test_fix.csv", index_col=0)["Path"]

# retrieve age class from folder structure, also get current file path
def get_some_info(id: str) -> tuple[str, str]:
	if id.startswith("đông_") or id.startswith("thành_") or id.startswith("tú_") or id.startswith("tuyến_"):
		age_class = "kindergarten"
		new_id = "audio_MDD/" + age_class + "/" + id.replace("_", "/", 1) + ".wav"
	else:
		age_class = "primaryschool"
		new_id = "audio_MDD/" + age_class + "/" + id + ".wav"
	return age_class, new_id
full_df["age_class"], full_df["current_path"] = zip(*full_df["Path"].apply(get_some_info))

# new file path following huggingface docs
def new_file_path(id: str) -> str:
	if id in train_files.values:
		folder = "vietmdd/train/"
	elif id in valid_files.values:
		folder = "vietmdd/validation/"
	elif id in test_files.values:
		folder = "vietmdd/test/"
	else:
		print(id)
		return ""
	return folder + id + ".wav"
full_df["new_path"] = full_df["Path"].apply(new_file_path)

# re-organize
for _, r in tqdm(full_df.iterrows()):
	os.rename(r["current_path"], r["new_path"])

# orphaned files without transcription
list_orphan_kindergarten  = glob("*/*.wav", root_dir="audio_MDD/kindergarten")  # on windows: [i.replace("\\", "/") for i in …]
list_orphan_primaryschool = glob("*.wav",   root_dir="audio_MDD/primaryschool")

rows_list = []
for row in list_orphan_kindergarten:
	new_path = "vietmdd/orphan/" + row.replace("/", "_")
	os.rename("audio_MDD/kindergarten/" + row, new_path)
	rows_list.append({
		"Path": "", "current_path": "",  # don’t need these info
		"Canonical": " ", "Transcript": " ",  # DO NOT LET empty string, mess up load_dataset later
		"age_class": "kindergarten", "new_path": new_path
	})
for row in list_orphan_primaryschool:
	new_path = "vietmdd/orphan/" + row
	os.rename("audio_MDD/primaryschool/" + row, new_path)
	rows_list.append({
		"Path": "", "current_path": "",  # don’t need these info
		"Canonical": " ", "Transcript": " ",  # DO NOT LET empty string, mess up load_dataset later
		"age_class": "primaryschool", "new_path": new_path
	})

yolo = (pd
	.concat([full_df, pd.DataFrame(rows_list)])
	.rename(columns={"new_path": "file_name", "Transcript": "observed_transcription", "Canonical": "original_text"})
)[["file_name", "observed_transcription", "original_text", "age_class"]]  # formatted following huggingface docs
yolo["file_name"] = yolo["file_name"].str.slice(start=len("vietmdd/"))

## huggingface datasets version < 3.4
# yolo.to_csv("vietmdd/metadata.csv", index=False, quoting=1)

## huggingface datasets version ≥ 3.4
yolo[["set", "file_name"]] = yolo["file_name"].str.split("/", n=1, expand=True)
yolo[yolo["set"] == "train"     ].drop(columns=["set"]).to_csv("vietmdd/train/metadata.csv",      index=False, quoting=1)
yolo[yolo["set"] == "test"      ].drop(columns=["set"]).to_csv("vietmdd/test/metadata.csv",       index=False, quoting=1)
yolo[yolo["set"] == "validation"].drop(columns=["set"]).to_csv("vietmdd/validation/metadata.csv", index=False, quoting=1)
yolo[yolo["set"] == "orphan"    ].drop(columns=["set"]).to_csv("vietmdd/orphan/metadata.csv",     index=False, quoting=1)

###############################################################################

from datasets import load_dataset
dataset = load_dataset("audiofolder", data_dir="vietmdd")
dataset.push_to_hub("doof-ferb/VietMDD", token="███")
#  dataset["orphan"] = load_dataset("audiofolder", data_dir="vietmdd/orphan", split='train')