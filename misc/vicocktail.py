#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert ViCocktail set to huggingface video datasets format

push_to_hub does not upload videos
https://github.com/huggingface/datasets/issues/7493

exceeds hf limits (>10k files per folder and >100k files in total)
save as either parquet or webdataset and then use upload_folder
tried `ds.to_parquet("…")` but the parquet file also doesnt contain video, as of `datasets` v4.0
the only workaround for my case is webdataset

prepare at least 350 GiB to download + unpack + tar
need more storage-efficient way to work
"""

###############################################################################

# %pip install -qU 'huggingface_hub[hf_xet]' 'datasets[audio,vision]' av

###############################################################################

import os, json, tarfile, datasets as hugDS
from tqdm import tqdm

ds = hugDS.load_dataset("nguyenvulebinh/ViCocktail")

os.makedirs("raw", exist_ok=True)
os.makedirs("tar", exist_ok=True)

N_FILES = 2000 # just a number bigger than 1167 (number of files in test set) can be 1200, 1500 or whatever
# put 2000 video files (+2000 .json files) per .tar files (uncompressed)

for split, data in ds.items():
	os.makedirs(f"raw/{split}", exist_ok=True)
	os.makedirs(f"tar/{split}", exist_ok=True)

	counter_tar = -1 # because counter get increment when index = 0
	tar_file = tarfile.open(f"tar/{split}/{counter_tar:03d}.tar","w")

	for i, sample in enumerate(tqdm(data)):

		if i % N_FILES == 0:
			tar_file.close()
			counter_tar += 1
			tar_file_path = f"tar/{split}/{counter_tar:03d}.tar"
			tqdm.write("start creating: " + tar_file_path) # print mess up tqdm
			tar_file = tarfile.open(tar_file_path, "w")

		vid_id = sample["__key__"]
		metadata = {
			"sample_id": str(sample["sample_id"], encoding="utf-8"),
			"transcription": str(sample["label"], encoding="utf-8").replace('"','')
		}
		vid_file_path = f"raw/{split}/{vid_id}.mp4"
		meta_file_path = f"raw/{split}/{vid_id}.json"

		with open(vid_file_path, "wb") as vid_file:
			vid_file.write(sample["video"])
		with open(meta_file_path, "w", encoding="utf-8") as meta_file:
			json.dump(metadata, meta_file, ensure_ascii=False, indent="\t")

		tar_file.add(vid_file_path, arcname=f"{vid_id}.mp4") # arcname to discard path info
		tar_file.add(meta_file_path, arcname=f"{vid_id}.json")

	tar_file.close()
	os.remove(f"tar/{split}/-01.tar")

###############################################################################

from huggingface_hub import HfApi
api = HfApi(token="███")
api.upload_large_folder(folder_path="tar", repo_id="doof-ferb/ViCocktail", repo_type="dataset")
