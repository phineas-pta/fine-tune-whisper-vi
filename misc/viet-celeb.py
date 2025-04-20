#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""convert Vietnam-Celeb set to huggingface audio datasets format"""

###############################################################################
# commands to download files in colab

# !gdown 1pMuT3DFzSwib7SVcRS8VkDwPuLTsemSG
# !gdown 1xayHt2HRqE1aJ4HvtUT40_9XlgvfDfRY
# !gdown 1MIlM78EbN_J9cApkNes_2_BrFrf8XwMc
# !gdown 1h6Na58DC03p-502B9QpC5Z_FadUwAdNA

# !zip -s- vietnam-celeb-part.zip -O full-dataset.zip
# !unzip full-dataset.zip
# !rm full-dataset.zip vietnam-celeb-part.z*
# !mkdir -p vietnam-celeb/{train,test}

# %pip install -qU 'datasets[audio]'

###############################################################################

import os
from glob import glob
from tqdm import tqdm

# retrieve files list for train & test sets
list_all = set(glob("*/*.wav", root_dir="data"))  # on windows: [i.replace("\\", "/") for i in …]
with open("vietnam-celeb-t.txt") as f:
	list_train = f.read().replace("\t", "/").splitlines()
list_train_bis = set(list_train) & list_all  # some files in list but non-existent
list_test = list_all - list_train_bis

for f in tqdm(list_train_bis):
	os.rename("data/" + f, "vietnam-celeb/train/" + f.replace("/", "_"))
for f in tqdm(list_test):
	os.rename("data/" + f, "vietnam-celeb/test/"  + f.replace("/", "_"))

###############################################################################

from datasets import load_dataset
dataset = load_dataset("audiofolder", data_dir="vietnam-celeb", drop_labels=True).map(
	lambda batch: {"original_file_path": batch["audio"]["path"][-len("id00000_00000.wav"):].replace("_", "/")},
	num_proc=4  # even 1 process already crash colab so run this scr
)
# dataset["train"][0]  # check
dataset.push_to_hub("doof-ferb/Vietnam-Celeb", token="███")
