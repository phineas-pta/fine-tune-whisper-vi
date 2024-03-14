#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert LSVSC_100 to huggingface audio datasets format"""

###############################################################################

import os, json
from tqdm import tqdm

with open("LSVSC_train.json", mode="r", encoding="utf-8") as f:
	train_table = json.load(f)
with open("LSVSC_test.json", mode="r", encoding="utf-8") as f:
	test_table = json.load(f)
with open("LSVSC_valid.json", mode="r", encoding="utf-8") as f:
	val_table = json.load(f)

train_list = [v["wav"] for v in train_table.values()]
test_list  = [v["wav"] for v in  test_table.values()]
val_list   = [v["wav"] for v in   val_table.values()]
assert set(train_list).intersection(set(test_list)) == set()
assert set(train_list).intersection(set(val_list)) == set()
assert set(test_list).intersection(set(val_list)) == set()


for f in tqdm(train_list):
	os.rename(f"data/{f}", f"data/train/{f}")
for f in tqdm(test_list):
	os.rename(f"data/{f}", f"data/test/{f}")
for f in tqdm(val_list):
	os.rename(f"data/{f}", f"data/validation/{f}")

###############################################################################

import pandas as pd

df = pd.read_json("helck_finale.json", orient="records")

file_list = {
	**{f: f"data/train/{f}"      for f in train_list},
	**{f: f"data/test/{f}"       for f in  test_list},
	**{f: f"data/validation/{f}" for f in   val_list}
}
df["file_name"] = df["wav"].apply(lambda f: file_list[f])

# a lot of anomalies in transcription
df["transcription"] = (df["text"]
	.str.replace(" ?- ?", " ", regex=True)
	.str.replace("\n+",   " ", regex=True)
	.str.replace("\r+",   " ", regex=True)
	.str.replace(" +",    " ", regex=True)
	.str.replace("\\", "")
	.str.replace(chr(65279), "")
	.str.strip()
)

topic_list = {
	"0": "news",
	"1": "movies, drama (dialogue)",
	"2": "stories, audio book",
	"3": "cultural issue",
	"4": "tourism, exploring",
	"5": "living viewpoints, living expats",
	"6": "sports",
	"7": "security related",
	"8": "traffic",
	"9": "healthcare",
}
gender_list = {"M": "male", "F": "female"}
dialect_list = {
	"0": "northern dialect",
	"1": "central dialect",
	"2": "highland central dialect",
	"3": "southern dialect",
	"4": "minority ethnic group dialect",
}
emotion_list = {
	"N": "neutral",
	"J": "joyful",
	"A": "angry",
	"S": "sad",
	"T": "tired",
	"D": "disgust",
	"F": "fear",
	"H": "happy",
}
age_list = {
	"C": "children",
	"Y": "young",
	"M": "middle age",
	"O": "old",
	"S": "very old",
}
def categorize(txt: str) -> pd.Series:
	if txt is None:
		res = [pd.NA] * 5
	else:
		yolo = list(txt)[1:6]
		res = [
			  topic_list.get(yolo[0], pd.NA),
			 gender_list.get(yolo[1], pd.NA),
			dialect_list.get(yolo[2], pd.NA),
			emotion_list.get(yolo[3], pd.NA),
			    age_list.get(yolo[4], pd.NA),
		]
	return pd.Series(res)

df[["topic", "gender", "dialect", "emotion", "age"]] = df["class"].apply(categorize)

df.drop(columns=["wav", "class", "duration", "text"], inplace=True)
df.to_csv("metadata.csv", index=False, quoting=1)

###############################################################################

from datasets import load_dataset
dataset = load_dataset("audiofolder", data_dir=".")
dataset.push_to_hub("doof-ferb/LSVSC", token="███")
