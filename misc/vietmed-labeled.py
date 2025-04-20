#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert VietMed labeled set to huggingface audio datasets format"""

###############################################################################
# commands to download files in colab

# those zip files contain quite weird file path inside
# in case error with gdown, see https://stackoverflow.com/a/67550427/10805680

# !gdown --folder 1WlL4gadQyCXnP8KuE9N-R9iveoC0kdt5  # folder with 10 zip files in official google drive download link
# !for file in labeled_medical_data/*.zip; do unzip -q "$file" && rm "$file"; done
# !mkdir -p vietmed/{train,test,validation}
# %pip install -qU 'datasets[audio]'

###############################################################################

import ast, pandas as pd
from tqdm import tqdm

# transcription files are json but badly formatted, need some processing
def read_file(file):
	with open(file, mode="r", encoding="utf8") as f:
		tmp0 = f.read()
	tmp1 = ast.literal_eval(tmp0)  # because file use single quotes, incompatible with JSON double quotes
	tmp2 = pd.DataFrame.from_records(tmp1)
	tmp2.drop(columns=["duration"], inplace=True)
	tmp2["seq_name"] = tmp2["seq_name"].str.split("/").str[-1] + ".ogg"  # file name with extension
	return tmp2

train_df = read_file("labeled_medical_data/labeled_medical_data_train_transcript.txt")
dev_df   = read_file("labeled_medical_data/labeled_medical_data_dev_transcript.txt")
test_df  = read_file("labeled_medical_data/labeled_medical_data_test_transcript.txt")
cv_df    = read_file("labeled_medical_data/labeled_medical_data_cv_transcript.txt")

# current file path
train_df["file"] = "train_audio/" + train_df["file"]
dev_df[  "file"] =   "dev_audio/" +   dev_df["file"]
test_df[ "file"] =  "test_audio/" +  test_df["file"]
cv_df[   "file"] =    "cv_audio/" +    cv_df["file"]

# new file path following huggingface docs
train_df["seq_name"] = "vietmed/train/"      + train_df["seq_name"]
dev_df[  "seq_name"] = "vietmed/validation/" +   dev_df["seq_name"]
test_df[ "seq_name"] = "vietmed/test/"       +  test_df["seq_name"]
cv_df[   "seq_name"] = "vietmed/train/"      +    cv_df["seq_name"]

# re-organize
yolo = pd.concat([train_df, cv_df, dev_df, test_df], axis=0)
for _, r in tqdm(yolo.iterrows()):
	os.rename(r["file"], r["seq_name"])

swag = (yolo
	.rename(columns={"seq_name": "file_name", "text": "transcription", "speaker_name": "Speaker ID"})
	.drop(columns=["file"])
)[["file_name", "transcription", "Speaker ID"]]
swag["file_name"] = swag["file_name"].str.slice(start=len("vietmed/"))
swag.to_csv("vietmed/metadata.csv", index=False, quoting=1)  # formatted following huggingface docs

###############################################################################

from datasets import load_dataset
dataset = load_dataset("audiofolder", data_dir="vietmed")
dataset.push_to_hub("doof-ferb/VietMed_labeled", token="███")

# !zip -r yolo.zip vietmed
