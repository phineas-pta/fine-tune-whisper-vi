#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BE AWARE: the huggingface docker image use python 3.8

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="my whisper training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-pretrained-model", default="vinai/PhoWhisper-medium", choices=["vinai/PhoWhisper-large", "vinai/PhoWhisper-medium", "openai/whisper-large-v3", "openai/whisper-medium"])
	parser.add_argument("-use-ytb-data", action="store_true", help="include youtube data to training set")
	parser.add_argument("-batch-size", type=int, default=8, help="should be multiple of 8")
	time_grp = parser.add_mutually_exclusive_group()
	time_grp.add_argument("-num-epochs", type=float, default=1.)
	time_grp.add_argument("-num-steps",  type=int, help="1 epoch ≈ 86k samples without youtube data, or ≈ 1.5M samples with ytb data")
	parser.add_argument("-save-path", default="./save")
	parser.add_argument("-resume-training", action="store_true")
	return parser.parse_args()

ARGS = parse_args()

###############################################################################

from dataclasses import dataclass
import torch
import datasets as hugDS
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, BitsAndBytesConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer
from accelerate import Accelerator
import peft

has_bf16 = torch.cuda.is_bf16_supported()  # GPU Ampere or later
accelerator = Accelerator(project_dir=ARGS.save_path, log_with="tensorboard", mixed_precision="bf16" if has_bf16 else "fp16")

FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(ARGS.pretrained_model)
TOKENIZER = WhisperTokenizer.from_pretrained(ARGS.pretrained_model, language="vi", task="transcribe")
MODEL = WhisperForConditionalGeneration.from_pretrained(
	ARGS.pretrained_model, use_cache=False, device_map={"": accelerator.device},
	quantization_config=BitsAndBytesConfig(
		load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.bfloat16 if has_bf16 else torch.float16
	)
)
MODEL.config.forced_decoder_ids = None
MODEL.config.suppress_tokens = []

DUMMY_TOKEN = -100

MODEL_BIS = peft.get_peft_model(
	peft.prepare_model_for_kbit_training(MODEL, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}),
	peft.LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=.05, bias="none")
)
if accelerator.is_main_process:
	MODEL_BIS.print_trainable_parameters()  # 16 millions = 1% of 1.6 billions params of whisper large

###############################################################################
# prepare data

SAMPLING_RATE = 16_000
def load_my_data(streaming=True, **kwargs):
	return hugDS.load_dataset(**kwargs, split="train", trust_remote_code=True, streaming=streaming).cast_column("audio", hugDS.Audio(sampling_rate=SAMPLING_RATE))


if ARGS.use_ytb_data:
	MY_DATA = hugDS.concatenate_datasets([  # total: 1.5M samples
		load_my_data(path="doof-ferb/fpt_fosd"),  # 25.9k
		load_my_data(path="doof-ferb/infore1_25hours"),  # 14.9k
		load_my_data(path="doof-ferb/LSVSC").select_columns(["audio", "transcription"]),  # 45k
		load_my_data(path="quocanh34/viet_vlsp"),  # 171k
		load_my_data(path="linhtran92/viet_youtube_asr_corpus_v2").select_columns(["audio", "transcription"]),  # 195k
		load_my_data(path="doof-ferb/infore2_audiobooks"),  # 315k
		load_my_data(path="linhtran92/viet_bud500"),  # 634k
	])
else:
	MY_DATA = hugDS.concatenate_datasets([  # total: 86k samples
		load_my_data(path="doof-ferb/fpt_fosd", streaming=False),  # 25.9k
		load_my_data(path="doof-ferb/infore1_25hours", streaming=False),  # 14.9k
		load_my_data(path="doof-ferb/LSVSC", streaming=False).select_columns(["audio", "transcription"]),  # 45k
	])


def prepare_dataset(batch):
	audio = batch["audio"]
	batch["input_length"] = len(audio["array"])  # compute input length
	batch["input_features"] = FEATURE_EXTRACTOR(audio["array"], sampling_rate=SAMPLING_RATE).input_features[0]  # compute log-Mel input features
	batch["labels"] = TOKENIZER(batch["transcription"]).input_ids  # encode target text to label ids
	batch["labels_length"] = len(batch["labels"])  # compute labels length
	return batch

def filter_inputs(input_length):
	"""Filter inputs with zero input length or longer than 30s"""
	return 0 < input_length < 48e4  # 30s × 16kHz

def filter_labels(labels_length):
	"""Filter label sequences longer than max length 448 tokens"""
	return labels_length < 448  # MODEL.config.max_length

MY_DATA = (MY_DATA
	.map(prepare_dataset)  # no `num_proc` coz streaming
	.filter(filter_inputs, input_columns= ["input_length"])
	.filter(filter_labels, input_columns=["labels_length"])
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
	def __call__(self, features):
		# split inputs and labels since they have to be of different lengths and need different padding methods
		input_features = [{"input_features": feature["input_features"]} for feature in features]
		label_features = [{"input_ids"     : feature["labels"]        } for feature in features]  # get the tokenized label sequences

		batch = FEATURE_EXTRACTOR.pad(input_features, return_tensors="pt")  # treat the audio inputs by simply returning torch tensors
		labels_batch =  TOKENIZER.pad(label_features, return_tensors="pt")  # pad the labels to max length
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), DUMMY_TOKEN)  # replace padding with -100 to ignore loss correctly

		if (labels[:, 0] == TOKENIZER.bos_token_id).all().cpu().item():  # if bos token is appended in previous tokenization step,
			labels = labels[:, 1:]  # cut bos token here as it’s append later anyways

		batch["labels"] = labels
		return batch

DATA_COLLATOR = DataCollatorSpeechSeq2SeqWithPadding()

###############################################################################
# training setup

# a practical learning rate while fine-tuning is a value 40× smaller than original used for pre-training
if "tiny" in ARGS.pretrained_model:
	LEARNING_RATE = 3.75e-5
elif "base" in ARGS.pretrained_model:
	LEARNING_RATE = 2.5e-5
elif "small" in ARGS.pretrained_model:
	LEARNING_RATE = 1.25e-5
elif "medium" in ARGS.pretrained_model:
	LEARNING_RATE = 6.25e-6
elif "large" in ARGS.pretrained_model:
	LEARNING_RATE = 5e-6
else:
	LEARNING_RATE = 5e-5


TRAINING_ARGS = Seq2SeqTrainingArguments(
	output_dir=ARGS.save_path,
	per_device_train_batch_size=ARGS.batch_size,
	# per_device_eval_batch_size=batch_size,
	fp16=not has_bf16,
	bf16=has_bf16, tf32=has_bf16,
	# torch_compile=True,  # SDPA not support whisper yet
	report_to=["tensorboard"],

	num_train_epochs=ARGS.num_epochs,
	max_steps=ARGS.num_steps,
	logging_steps=25,
	save_steps=50,
	evaluation_strategy="no",
	save_total_limit=5,
	accelerator_config={"split_batches": True},  # mandatory for streaming datasets

	optim="adamw_bnb_8bit",  # 8-bit AdamW optimizer: lower vram usage than default AdamW
	learning_rate=LEARNING_RATE,
	warmup_steps=.05,  # keep between 5-15%
	gradient_accumulation_steps=1 if ARGS.batch_size >= 8 else 8 // ARGS.batch_size,  # keep effective batch size as min 8 per device
	remove_unused_columns=False, label_names=["labels"],  # required by PEFT
	# predict_with_generate=True,  # must disable coz PEFT
)

TRAINER = Seq2SeqTrainer(
	args=TRAINING_ARGS,
	model=MODEL_BIS,
	train_dataset=MY_DATA,
	data_collator=DATA_COLLATOR,
	# compute_metrics=compute_metrics,  # must disable coz PEFT
	tokenizer=FEATURE_EXTRACTOR,  # not TOKENIZER
)

TRAINER.train(resume_from_checkpoint=ARGS.resume_training)

accelerator.wait_for_everyone()
if accelerator.is_main_process:
	TRAINER.save_model()
