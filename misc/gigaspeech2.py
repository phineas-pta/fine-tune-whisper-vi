#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert GigaSpeech 2 to huggingface audio datasets format
ATTENTION: do not run this file directly, see instructions below
"""

###############################################################################
# commands to run files in colab

# !mkdir gigaspeech2
# !touch gigaspeech2/README.md
# !touch gigaspeech2/gigaspeech2.py
# then copy the file content to gigaspeech2.py

# %pip install -qU 'datasets[audio]'
# import datasets as hugDS
# SAMPLING_RATE = 16_000
# yolo = hugDS.load_dataset("gigaspeech2", streaming=True, trust_remote_code=True)  # streaming to avoid download ~1TB
# ds = hugDS.DatasetDict()
# ds["train"]      = hugDS.Dataset.from_generator(yolo["train"     ].__iter__)
# ds["validation"] = hugDS.Dataset.from_generator(yolo["validation"].__iter__)
# ds["test"]       = hugDS.Dataset.from_generator(yolo["test"      ].__iter__)
# ds = ds.cast_column("audio", hugDS.Audio(sampling_rate=SAMPLING_RATE))  # take back column type
# ds.push_to_hub("doof-ferb/gigaspeech2_vie", token="███")

###############################################################################

import os
from huggingface_hub import hf_hub_download
import datasets as hugDS

train_file = hf_hub_download(repo_type="dataset", repo_id="speechcolab/gigaspeech2", filename="data/vi/train_raw.tsv")
dev_file   = hf_hub_download(repo_type="dataset", repo_id="speechcolab/gigaspeech2", filename="data/vi/dev.tsv")
test_file  = hf_hub_download(repo_type="dataset", repo_id="speechcolab/gigaspeech2", filename="data/vi/test.tsv")

def read_tsv(file: str) -> dict[str, str]:
	table = {}  # {"filename": "transcription"}
	with open(file, mode="r", encoding="utf8") as f:
		for line in f:
			Giga = line.split("\t")
			table[Giga[0]] = Giga[1].strip("\n").lower()
	return table

train_table = read_tsv(train_file)
dev_table   = read_tsv(dev_file)
test_table  = read_tsv(test_file)


_N = 240
_URL = "https://huggingface.co/datasets/speechcolab/gigaspeech2/resolve/main/data/vi"
_URLS = {"dev": _URL + "/dev.tar.gz", "test": _URL + "/test.tar.gz"}
# _URLS["train"] = list(…) throw error with DownloadManager later
for i in range(_N):
	_URLS[f"train{i}"] = f"{_URL}/train/{i}.tar.gz"


class GigaConfig(hugDS.BuilderConfig):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class Giga(hugDS.GeneratorBasedBuilder):
	BUILDER_CONFIGS = [GigaConfig(name="plain_text", version=hugDS.Version("1.0.0"), description="Plain text")]

	def _info(self):
		return hugDS.DatasetInfo(
			description="placeholder",
			features=hugDS.Features({"audio": hugDS.Audio(sampling_rate=16000), "transcription": hugDS.Value("string")}),
			supervised_keys=None,
			task_templates=[hugDS.tasks.AutomaticSpeechRecognition()],
		)

	def _split_generators(self, dl_manager: hugDS.DownloadManager):
		archive = dl_manager.download(_URLS)  # cannot use download_and_extract in streaming mode
		yolo = [  # gen_kwargs will be passed to _generate_examples below
			hugDS.SplitGenerator(name=hugDS.Split.VALIDATION, gen_kwargs={
				"transcript_table": dev_table,
				"audio_files": dl_manager.iter_archive(archive["dev"])
			}),
			hugDS.SplitGenerator(name=hugDS.Split.TEST, gen_kwargs={
				"transcript_table": test_table,
				"audio_files": dl_manager.iter_archive(archive["test"])
			}),
		]
		for i in range(_N):
			yolo.append(
				hugDS.SplitGenerator(name=hugDS.Split.TRAIN, gen_kwargs={
					"transcript_table": train_table,
					"audio_files": dl_manager.iter_archive(archive[f"train{i}"])
				})
			)
		return yolo

	def _generate_examples(self, transcript_table, audio_files):
		key = 0
		for path, f in audio_files:
			file_id = os.path.splitext(os.path.basename(path))[0]
			yield key, {
				"audio": {"path": path, "bytes": f.read()},
				"transcription": transcript_table[file_id],
			}
			key += 1
