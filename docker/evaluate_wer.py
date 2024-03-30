#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BE AWARE: the huggingface docker image use python 3.8

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="my whisper evaluation script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-save-path", default="./save")
	parser.add_argument("-batch-size", type=int, default=32, help="should be multiple of 8; for T4 @ float16: 16 for large model, 32 medium")
	return parser.parse_args()

ARGS = parse_args()


from tqdm import tqdm
import torch
from peft import PeftModel, PeftConfig
from transformers import AutomaticSpeechRecognitionPipeline, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import datasets as hugDS
import jiwer

JIWER_TRANS = jiwer.Compose([  # DO NOT use `jiwer.RemoveEmptyStrings` it can cause rows count mismatch
	jiwer.ToLowerCase(),
	jiwer.RemoveKaldiNonWords(),
	jiwer.RemoveMultipleSpaces(),
	jiwer.Strip(),
	jiwer.RemovePunctuation(),
	jiwer.ReduceToListOfListOfWords(),
])

SAMPLING_RATE = 16_000
def load_my_data(streaming=True, **kwargs):
	return hugDS.load_dataset(**kwargs, split="test", trust_remote_code=True, streaming=streaming).cast_column("audio", hugDS.Audio(sampling_rate=SAMPLING_RATE))

MY_DATA = hugDS.IterableDatasetDict()
MY_DATA["commonvoice"] = load_my_data(path="mozilla-foundation/common_voice_16_1", name="vi",  ).select_columns(["audio", "sentence"])
# MY_DATA["fleurs"] # disable FLEURS because error with tensor size mismatch when batching, see bottom for non-batched inference
MY_DATA["vivos"]       = load_my_data(path="vivos"                                             ).select_columns(["audio", "sentence"])
MY_DATA["bud500"]      = load_my_data(path="linhtran92/viet_bud500"                            ).rename_column("transcription", "sentence")
MY_DATA["lsvsc"]       = load_my_data(path="doof-ferb/LSVSC"                                   ).select_columns(["audio", "transcription"]).rename_column("transcription", "sentence")

ROWS_COUNT = {
	"commonvoice": 1326,
	"fleurs":       857,
	"vivos":        760,
	"bud500":      7500,
	"lsvsc":       5683,
}

BASE_MODEL_ID = PeftConfig.from_pretrained(ARGS.save_path).base_model_name_or_path
print("adapter to", BASE_MODEL_ID)

# declare task & language in extractor & tokenizer have no effect in inference
FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(BASE_MODEL_ID)
TOKENIZER = WhisperTokenizer.from_pretrained(BASE_MODEL_ID)

MODEL = PeftModel.from_pretrained(
	WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID, device_map="auto", torch_dtype=torch.float16),
	ARGS.save_path
).merge_and_unload(progressbar=True)  # reduce latency with LoRA

PIPE = AutomaticSpeechRecognitionPipeline(model=MODEL, tokenizer=TOKENIZER, feature_extractor=FEATURE_EXTRACTOR)
PIPE_KWARGS = {"language": "vi", "task": "transcribe"}


# workaround because KeyDataset(MY_DATA[split], "audio") raise error with streaming datasets
def data(batch):
	for row in batch:
		yield row["audio"]


@torch.autocast(device_type="cuda")  # required by PEFT
@torch.inference_mode()
def predict(split):
	batch = MY_DATA[split]
	y_pred = [out["text"] for out in tqdm(PIPE(data(batch), generate_kwargs=PIPE_KWARGS, batch_size=ARGS.batch_size), total=ROWS_COUNT[split], unit="samples", desc=f"{split=}")]
	y_true = [row["sentence"] for row in batch]
	assert len(y_pred) == len(y_true)
	return y_true, y_pred


for split in MY_DATA.keys():
	y_true, y_pred = predict(split)
	torch.cuda.empty_cache()  # forced clean
	wer = 100 * jiwer.wer(
		reference=y_true,
		hypothesis=y_pred,
		reference_transform=JIWER_TRANS,
		hypothesis_transform=JIWER_TRANS,
	)
	if 0 < wer < 100:
		print(f"WER on {split} = {wer:.1f}%", end="\n\n")
	else:
		print("something wrong, check 5 first & last transcription:")
		print(y_true[:5], y_true[-5:])
		print(y_pred[:5], y_pred[-5:])

###############################################################################
# evaluate on FLEURS in non-batched inference (very very slow)

data_fleurs = load_my_data(path="google/fleurs", name="vi_vn", streaming=False).select_columns(["audio", "transcription"])

@torch.autocast(device_type="cuda")  # required by PEFT
@torch.inference_mode()
def predict_fleurs(batch):
	batch["pred"] = PIPE(batch["audio"], generate_kwargs=PIPE_KWARGS)["text"]
	return batch
data_fleurs = data_fleurs.map(predict_fleurs)  # progress bar included

wer = 100 * jiwer.wer(
	reference=data_fleurs["transcription"],
	hypothesis=data_fleurs["pred"],
	reference_transform=JIWER_TRANS,
	hypothesis_transform=JIWER_TRANS,
)
print(f"WER on FLEURS = {wer:.1f}%", end="\n\n")
