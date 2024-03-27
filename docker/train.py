#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BE AWARE: the huggingface docker image use python 3.8

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description="my whisper training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("-pretrained-model", default="vinai/PhoWhisper-large", choices=["vinai/PhoWhisper-large", "openai/whisper-large-v2", "openai/whisper-large-v3"])
	parser.add_argument("-use-ytb-data", action="store_true", help="include youtube data to training set")
	parser.add_argument("-batch-size", type=int, default=8, help="should be multiple of 8")
	parser.add_argument("-total-steps", type=int, default=int(86000), help="1 epoch ≈ 86k steps without youtube data, or ≈ 1.5M steps with ytb data")
	# parser.add_argument("-bf16", action="store_true", help="enable optimizations for Ampere or later GPU")
	parser.add_argument("-resume-training", action="store_true", help="resume training from ./save")
	return parser.parse_args()

ARGS = parse_args()


from dataclasses import dataclass
import datasets as hugDS
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, BitsAndBytesConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer
import peft
import accelerate

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


FEATURE_EXTRACTOR = WhisperFeatureExtractor.from_pretrained(ARGS.pretrained_model)
TOKENIZER = WhisperTokenizer.from_pretrained(ARGS.pretrained_model, language="vi", task="transcribe")
MODEL = WhisperForConditionalGeneration.from_pretrained(ARGS.pretrained_model, use_cache=False, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
MODEL.config.forced_decoder_ids = None
MODEL.config.suppress_tokens = []

# setup to train on multi-GPU with PEFT, see: https://github.com/huggingface/peft/issues/242#issuecomment-1491447956
DEV_MAP = MODEL.hf_device_map.copy()
DEV_MAP["model.decoder.embed_tokens"] = DEV_MAP["model.decoder.embed_positions"] = DEV_MAP["proj_out"] = MODEL._hf_hook.execution_device
accelerate.dispatch_model(MODEL, device_map=DEV_MAP)
setattr(MODEL, "model_parallel", True)
setattr(MODEL, "is_parallelizable", True)

DUMMY_TOKEN = -100

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

MODEL_BIS = peft.get_peft_model(
	peft.prepare_model_for_kbit_training(MODEL, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}),
	peft.LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=.05, bias="none")
)
MODEL_BIS.print_trainable_parameters()  # 16 millions = 1% of 1.6 billions params of whisper large

SAVE_PATH = "./save"

TRAINING_ARGS = Seq2SeqTrainingArguments(
	output_dir=SAVE_PATH,
	per_device_train_batch_size=ARGS.batch_size,
	fp16=True,
	# bf16=ARGS.bf16, tf32=ARGS.bf16,  # not working properly with PEFT
	# torch_compile=ARGS.bf16,  # weird error with SDPA attention
	report_to=["tensorboard"],

	max_steps=ARGS.total_steps,
	logging_steps=25,
	save_steps=50,
	evaluation_strategy="no",
	# save_total_limit=3,

	learning_rate=1e-3,
	warmup_ratio=.05,  # keep between 5-15%
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
TRAINER.save_model()
