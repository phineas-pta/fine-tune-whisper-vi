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
# !wget -P gigaspeech2 https://raw.githubusercontent.com/phineas-pta/fine-tune-whisper-vi/main/misc/gigaspeech2.py
# %pip install -qU 'datasets[audio]'
# import datasets as hugDS
# yolo = hugDS.load_dataset("gigaspeech2", streaming=True, trust_remote_code=True)  # streaming to avoid download ~1TB immediately
# ds = hugDS.DatasetDict()
# ds["train"]      = hugDS.Dataset.from_generator(yolo["train"     ].__iter__)
# ds["validation"] = hugDS.Dataset.from_generator(yolo["validation"].__iter__)
# ds["test"]       = hugDS.Dataset.from_generator(yolo["test"      ].__iter__)
# ds = ds.cast_column("audio", hugDS.Audio(sampling_rate=16_000))  # take back column type
# ds.push_to_hub("doof-ferb/gigaspeech2_vie", token="███")

###############################################################################

from huggingface_hub import hf_hub_download
import datasets as hugDS
import os
from shutil import rmtree

_LOGGER = hugDS.utils.logging.get_logger(__name__)

_TRAIN_RAW_FILE = hf_hub_download(repo_type="dataset", repo_id="speechcolab/gigaspeech2", filename="data/vi/train_raw.tsv")
_TRAIN_REF_FILE = hf_hub_download(repo_type="dataset", repo_id="speechcolab/gigaspeech2", filename="data/vi/train_refined.tsv")
_DEV_FILE       = hf_hub_download(repo_type="dataset", repo_id="speechcolab/gigaspeech2", filename="data/vi/dev.tsv")
_TEST_FILE      = hf_hub_download(repo_type="dataset", repo_id="speechcolab/gigaspeech2", filename="data/vi/test.tsv")

def read_tsv(file: str) -> dict[str, str]:
	table = {}  # {"filename": "transcription"}
	with open(file, mode="r", encoding="utf8") as f:
		for line in f:
			yolo = line.split("\t")
			table[yolo[0]] = yolo[1].strip("\n").lower()
	return table

_TRAIN_RAW_FILE = read_tsv(_TRAIN_RAW_FILE)
_TRAIN_REF_FILE = read_tsv(_TRAIN_REF_FILE)
_DEV_TABLE      = read_tsv(_DEV_FILE)
_TEST_TABLE     = read_tsv(_TEST_FILE)


_N = 240  # vie subset = 240 (0→239), ind subset = 592, tha subset = 193
_URL = "https://huggingface.co/datasets/speechcolab/gigaspeech2/resolve/main/data/vi"
_URLS = {"dev": _URL + "/dev.tar.gz", "test": _URL + "/test.tar.gz"}
# _URLS["train"] = list(…) throw error with DownloadManager later
for i in range(_N):
	_URLS[f"train{i}"] = f"{_URL}/train/{i}.tar.gz"


class GigaSpeech2_Dataset(hugDS.GeneratorBasedBuilder):
	VERSION = hugDS.Version("1.0.0")

	def _info(self):
		return hugDS.DatasetInfo(
			description="placeholder",
			features=hugDS.Features({"audio": hugDS.Audio(sampling_rate=16000), "transcription": hugDS.Value("string")}),
			supervised_keys=None,
			task_templates=[hugDS.tasks.AutomaticSpeechRecognition()],
		)

	def _split_generators(self, dl_manager: hugDS.DownloadManager) -> list[hugDS.SplitGenerator]:
		archive = dl_manager.download(_URLS)  # cannot use download_and_extract in streaming mode
		return [  # gen_kwargs will be passed to _generate_examples below
			hugDS.SplitGenerator(name="train_raw", gen_kwargs={
				"split_name": "train_raw",
				"transcript_table": _TRAIN_RAW_FILE,
				"audio_files": [dl_manager.iter_archive(archive[f"train{i}"]) for i in range(_N)],
				# cannot use `itertools.chain` got TypeError: can't pickle generator objects
			}),
			hugDS.SplitGenerator(name="train_refined", gen_kwargs={
				"split_name": "train_refined",
				"transcript_table": _TRAIN_REF_FILE,
				"audio_files": [dl_manager.iter_archive(archive[f"train{i}"]) for i in range(_N)],
			}),
			hugDS.SplitGenerator(name="validation", gen_kwargs={
				"split_name": "validation",
				"transcript_table": _DEV_TABLE,
				"audio_files": [dl_manager.iter_archive(archive["dev"])],  # make a list same as train set above
			}),
			hugDS.SplitGenerator(name="test", gen_kwargs={
				"split_name": "test",
				"transcript_table": _TEST_TABLE,
				"audio_files": [dl_manager.iter_archive(archive["test"])],  # make a list same as train set above
			}),
		]

	def _generate_examples(self, split_name: str, transcript_table: dict, audio_files: list[tuple]) -> tuple[int, dict]:
		key = 0  # for legacy reason (tensorflow datasets)
		_LOGGER.info("processing: split: " + split_name)
		for files in audio_files:
			for path, f in files:
				file_id = os.path.splitext(os.path.basename(path))[0]
				transcript = transcript_table.get(file_id)
				if transcript is None:
					_LOGGER.warning("skipping " + filepath)
				else:
					yield key, {"audio": {"path": path, "bytes": f.read()}, "transcription": transcript}
					key += 1
			# try clean cache to prevent out of disk space on colab/kaggle BUT DOESN’T HELP
